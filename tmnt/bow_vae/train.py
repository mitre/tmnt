# coding: utf-8

import math
import logging
import datetime
import io
import os
import json
import mxnet as mx
import numpy as np
import pickle
import copy
import socket

from mxnet import autograd
from mxnet import gluon
import gluonnlp as nlp
from pathlib import Path

from tmnt.bow_vae.bow_doc_loader import DataIterLoader, collect_sparse_data, BowDataSet, collect_stream_as_sparse_matrix
from tmnt.bow_vae.bow_models import BowNTM, MetaDataBowNTM
from tmnt.bow_vae.topic_seeds import get_seed_matrix_from_file
from tmnt.utils.log_utils import logging_config
from tmnt.utils.mat_utils import export_sparse_matrix, export_vocab
from tmnt.utils.random import seed_rng
from tmnt.coherence.npmi import EvaluateNPMI
from tmnt.modsel.configuration import TMNTConfig

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB

__all__ = ['model_select_bow_vae', 'train_bow_vae']


def usedPort(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = True
    try:
        sock.bind(("0.0.0.0", port))
        result = False
    except:
        result = True
    sock.close()
    return result


def get_wd_freqs(data_csr, max_sample_size=1000000):
    sample_size = min(max_sample_size, data_csr.shape[0])
    data = data_csr[:sample_size] 
    sums = mx.nd.sum(data, axis=0)
    return sums


def evaluate(model, data_loader, total_words, args, ctx=mx.cpu(), debug=False):
    total_rec_loss = 0
    for i, (data,labels) in enumerate(data_loader):
        if labels is None:            
            labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
            labels = labels.as_in_context(ctx)
        data = data.as_in_context(ctx)
        _, kl_loss, rec_loss, _, _, _, log_out = model(data, labels) if args.use_labels_as_covars else model(data)
        total_rec_loss += rec_loss.sum().asscalar()
    perplexity = math.exp(total_rec_loss / total_words)
    logging.info("TEST/VALIDATION Perplexity = {}".format(perplexity))
    return perplexity


def compute_coherence(model, k, test_data, log_terms=False):
    w = model.decoder.collect_params().get('weight').data()
    num_topics = model.n_latent
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    num_topics = min(num_topics, sorted_ids.shape[-1])
    top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]
    npmi_eval = EvaluateNPMI(top_k_words_per_topic)
    npmi = npmi_eval.evaluate_csr_mat(test_data)
    unique_term_ids = set()
    unique_limit = 5  ## only consider the top 5 terms for each topic when looking at degree of redundancy
    for i in range(num_topics):
        topic_ids = list(top_k_words_per_topic[i][:unique_limit])
        for j in range(len(topic_ids)):
            unique_term_ids.add(topic_ids[j])
    redundancy = (1.0 - (float(len(unique_term_ids)) / num_topics / unique_limit)) ** 2
    logging.info("Test Coherence: {}".format(npmi))
    logging.info("Test Redundancy at 5: {}".format(redundancy))    
    if log_terms:
        top_k_tokens = [list(map(lambda x: model.vocabulary.idx_to_token[x], list(li))) for li in top_k_words_per_topic]
        for i in range(num_topics):
            logging.info("Topic {}: {}".format(i, top_k_tokens[i]))
    return npmi, redundancy


def analyze_seed_matrix(model, seed_matrix):
    w = model.decoder.collect_params().get('weight').data()
    ts = mx.nd.take(w, seed_matrix)   ## should have shape (T', S', T)
    ts_sums = mx.nd.sum(ts, axis=1)
    ts_probs = mx.nd.softmax(ts_sums)
    print("ts_prob = {}".format(ts_probs))
    entropies = -mx.nd.sum(ts_probs * mx.nd.log(ts_probs), axis=1)
    print("entropies = {}".format(entropies))
    seed_means = mx.nd.mean(ts, axis=1)  # (G,K)
    seed_pr = mx.nd.softmax(seed_means)
    per_topic_entropy_1 = -mx.nd.sum(seed_pr * mx.nd.log(seed_pr), axis=0)
    print("per_topic_entropy = {}".format(per_topic_entropy_1))
    total_topic_sum_means = mx.nd.sum(seed_means, axis=0)
    print("sum means = {}".format(total_topic_sum_means))    
    per_topic_entropy = mx.nd.sum(per_topic_entropy_1 * total_topic_sum_means)
    

def log_top_k_words_per_topic(model, vocab, num_topics, k):
    w = model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    for t in range(num_topics):
        top_k = [ vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
        term_str = ' '.join(top_k)
        logging.info("Topic {}: {}".format(str(t), term_str))


class BowVAEWorker(Worker):
    def __init__(self, c_args, vocabulary, data_train_csr, total_tr_words,
                 data_test_csr, total_tst_words, train_labels=None, test_labels=None, ctx=mx.cpu(), max_budget=27, **kwargs):
        super().__init__(**kwargs)
        self.c_args = c_args
        self.total_tr_words = total_tr_words
        self.total_tst_words = total_tst_words
        self.vocabulary = vocabulary
        self.ctx = ctx
        self.data_test_csr = data_test_csr
        self.max_budget = max_budget
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.data_train_csr = data_train_csr
        self.data_test_csr = data_test_csr
        
        self.search_mode = False
        
        self.wd_freqs = get_wd_freqs(data_train_csr)
        self.vocab_cache = {}
        
        self.seed_matrix = None
        if c_args.topic_seed_file:
            self.seed_matrix = get_seed_matrix_from_file(c_args.topic_seed_file, vocabulary)


    def _initialize_embedding_layer(self, embedding_source, config):
        """Initialize the embedding layer randomly or using pre-trained embeddings provided
        
        Parameters
        ----------
        embedding_source: string denoting a Gluon-NLP embedding source with the format <type>:<name> where <type>
        is 'glove', 'fasttext' or 'word2vec' and <name> denotes the embedding name (e.g. 'glove.6B.300d').
        See `gluonnlp.embedding.list_sources()` for a full list
        config: `Configuration` for this model evaluation run

        Returns
        -------
        vocab: Resulting GluonNLP vocabulary with initialized embedding
        emb_size: Size of embedding (based on pre-trained embedding or specified)
        """
        vocab = self.vocabulary
        if embedding_source and embedding_source != 'random':
            if self.vocab_cache.get(embedding_source):
                vocab = copy.deepcopy(self.vocab_cache[embedding_source])
            else:
                e_type, e_name = tuple(embedding_source.split(':'))
                pt_embedding = nlp.embedding.create(e_type, source=e_name)
                vocab = copy.deepcopy(self.vocabulary) ## create a copy of the vocab to attach the vocab to 
                vocab.set_embedding(pt_embedding)
                self.vocab_cache[embedding_source] = copy.deepcopy(vocab) ## cache another copy so the pre-trained embedding is preserverd
            emb_size = len(vocab.embedding.idx_to_vec[0])
            num_oov = 0
            for word in vocab.embedding._idx_to_token:
                if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                    logging.info("Term {} is OOV".format(word))
                    num_oov += 1
                    vocab.embedding[word] = mx.nd.random.normal(0, 1.0, emb_size)
            logging.info(">> {} Words did not appear in embedding source {}".format(num_oov, embedding_source))
        else:
            vocab.set_embedding(None) ## unset embedding
            emb_size = int(config['embedding_size'])
        return vocab, emb_size


    def _update_details(self, details, elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss):
        """Update loss details during training for logging and analysis
        """
        details['kl_loss']  += kl_loss.sum().asscalar()
        details['l1_pen']   += l1_pen.sum().asscalar()
        details['rec_loss'] += rec_loss.sum().asscalar()
        if coherence_loss is not None:
            details['coherence_loss'] += coherence_loss.sum().asscalar()
        if entropies is not None:
            details['entropies_loss'] += entropies.sum().asscalar()
        details['epoch_loss'] += elbo.sum().asscalar()

    def _log_details(self, details, epoch):
        """Log accumulated details (e.g. loss values) for a given epoch
        
        Parameters
        ----------
        details: dictionary - with various details (loss values) to keep track of
        epoch: int - current epoch number
        """
        tr_size = details['tr_size']
        nd = {}
        for (k,v) in details.items():
            nd[k] = v / tr_size
        logging.info("Epoch {}: Loss = {}, [ KL loss = {:8.4f} ] [ L1 loss = {:8.4f} ] [ Rec loss = {:8.4f}] [ Coherence loss = {:8.4f} ] [ Entropy losss = {:8.4f} ]".
                     format(epoch, nd['epoch_loss'], nd['kl_loss'], nd['l1_pen'], nd['rec_loss'], nd['coherence_loss'], nd['entropies_loss']))

    def _get_model(self, config):
        """Take a model configuration - specified by a config file or as determined by model selection and 
        return a VAE topic model ready for training.

        Parameters
        ----------
        config: a `ConfigSpace` object
        """
        
        lr = config['lr']
        latent_distrib = config['latent_distribution']
        optimizer = config['optimizer']
        n_latent = int(config['n_latent'])
        kappa = float(config.get('kappa', 100.0))
        enc_hidden_dim = int(config['enc_hidden_dim'])
        target_sparsity = float(config.get('target_sparsity', 0.0))
        coherence_reg_penalty = float(config.get('coherence_regularizer_penalty', 0.0))
        batch_size = int(config['batch_size'])

        l1_coef = self.c_args.init_sparsity_pen if target_sparsity > 0.0 else 0.0
        embedding_source = config.get('embedding_source')
        fixed_embedding = config.get('fixed_embedding') == 'True'
        vocab, emb_size = self._initialize_embedding_layer(embedding_source, config)
        
        if self.c_args.use_labels_as_covars and self.train_labels is not None:
            n_covars = mx.nd.max(self.train_labels).asscalar() + 1
            self.train_labels = mx.nd.one_hot(self.train_labels, n_covars)
            self.test_labels = mx.nd.one_hot(self.test_labels, n_covars) if self.test_labels is not None else None
            model = \
                MetaDataBowNTM(n_covars, vocab, enc_hidden_dim, n_latent, emb_size,
                               fixed_embedding=fixed_embedding, latent_distrib=latent_distrib, kappa=kappa,
                               init_l1=l1_coef, coherence_reg_penalty=coherence_reg_penalty, 
                               batch_size=batch_size, wd_freqs=self.wd_freqs, ctx=self.ctx)
        else:
            model = \
                BowNTM(vocab, enc_hidden_dim, n_latent, emb_size,
                   fixed_embedding=fixed_embedding, latent_distrib=latent_distrib,
                       init_l1=l1_coef, coherence_reg_penalty=coherence_reg_penalty, target_sparsity=target_sparsity, kappa=kappa,
                   batch_size=batch_size, wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, ctx=self.ctx)
        trainer = gluon.Trainer(model.collect_params(), optimizer, {'learning_rate': lr})
        if self.c_args.hybridize:
            model.hybridize()
        return model, trainer

    def _eval_trace(self, model, epoch, test_dataloader):
        """Evaluate the model against test/validation data and optionally write to a trace file.
        
        Parameters
        ----------
        model: VAE model
        epoch: int - the current epoch
        """
        if test_dataloader is not None and (epoch + 1) % self.c_args.eval_freq == 0:
            perplexity = evaluate(model, test_dataloader, self.total_tst_words, self.c_args, self.ctx)
            if self.c_args.trace_file:
                otype = 'a+' if epoch >= self.c_args.eval_freq else 'w+'
                with io.open(self.c_args.trace_file, otype) as fp:
                    if otype == 'w+':
                        fp.write("Epoch,PPL,NPMI\n")
                    npmi,_ = compute_coherence(model, 10, self.data_test_csr, log_terms=True)
                    fp.write("{:3d},{:10.2f},{:8.4f}\n".format(epoch, perplexity, npmi))

    def _l1_regularize(self, model, cur_l1_coef):
        """Apply a regularization term based on magnitudes of the decoder (topic-term) weights.
        Set the L1 coeffficient based on these magnitudes which will be used to compute L1 loss term

        Parameters
        ----------
        model: VAE model
        """
        dec_weights = model.decoder.collect_params().get('weight').data().abs()
        ratio_small_weights = (dec_weights < self.c_args.sparsity_threshold).sum().asscalar() / dec_weights.size
        l1_coef = cur_l1_coef * math.pow(2.0, model.target_sparsity - ratio_small_weights)
        logging.info("Setting L1 coeffficient to {} [sparsity ratio = {}]".format(l1_coef, ratio_small_weights))
        model.l1_pen_const.set_data(mx.nd.array([l1_coef]))
        return l1_coef

    def _train_model(self, config, budget):
        """Main training function which takes a single model configuration and a budget (i.e. number of epochs) and
        fits the model to the training data.
        
        Parameters
        ----------
        config: `Configuration` object within the specified `ConfigSpace`
        budget: int - Number of iterations to use when building the model

        Returns
        -------
        model: VAE model with trained parameters
        res: Result dictionary with details of training run for use in model selection
        """
        logging.info("Evaluating with Budget {} against Config: {}".format(budget, config))
        model, trainer = self._get_model(config)
        batch_size = int(config['batch_size'])
        l1_coef = self.c_args.init_sparsity_pen
        train_dataloader = \
            DataIterLoader(mx.io.NDArrayIter(self.data_train_csr, self.train_labels, batch_size, last_batch_handle='discard', shuffle=True))
        test_dataloader = \
            DataIterLoader(mx.io.NDArrayIter(self.data_test_csr, self.test_labels, batch_size, last_batch_handle='discard', shuffle=True))

        for epoch in range(int(budget)):
            details = {'epoch_loss': 0.0, 'rec_loss': 0.0, 'l1_pen': 0.0, 'kl_loss': 0.0,
                       'entropies_loss': 0.0, 'coherence_loss': 0.0, 'tr_size': 0.0}
            for i, (data, labels) in enumerate(train_dataloader):
                details['tr_size'] += data.shape[0]
                if labels is None or labels.size == 0:
                    labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
                    labels = labels.as_in_context(self.ctx)
                data = data.as_in_context(self.ctx)
                with autograd.record():
                    elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss, _ = \
                        model(data, labels) if self.c_args.use_labels_as_covars else model(data)
                    elbo_mean = elbo.mean()
                elbo_mean.backward()
                trainer.step(data.shape[0]) 
                self._update_details(details, elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss)
            self._log_details(details, epoch)
            if not self.search_mode:
                self._eval_trace(model, epoch, test_dataloader)
            if model.target_sparsity > 0.0:
                l1_coef = self._l1_regularize(model, l1_coef)
        mx.nd.waitall()
        perplexity = evaluate(model, test_dataloader, self.total_tst_words, self.c_args, self.ctx)
        npmi, redundancy = compute_coherence(model, 10, self.data_test_csr, log_terms=True)
        try:
            coherence_coefficient = self.c_args.coherence_coefficient
        except AttributeError:
            coherence_coefficient = 1.0
        res = {
            'loss': 1.0 - (npmi * coherence_coefficient) + redundancy + (perplexity / 1000),
            'info': {
                'test_perplexity': perplexity,
                'redundancy': redundancy,
                'test_npmi': npmi
            }
        }
        return model, res


    def compute(self, config, budget, working_directory, *args, **kwargs):
        """Worker method used within HPBandSter model selection.
        
        Parameters
        ---------
        config: `Configuration` to use to train/evaluate the model
        budget: int - number of iterations to train
        working_directory: string - optional directory (not used by this worker)

        Returns
        -------
        Result dictionary for use in model selection
        """
        _, res = self._train_model(config, budget)
        return res

    def retrain_best_config(self, config, budget):
        """Train a model as per the provided `Configuration` and `budget` and write to file.
        
        Parameters
        ----------
        config: `Configuration` to use to train/evaluate the model
        budget: int - number of iterations to train
        """
        model, _ = self._train_model(config, budget)
        write_model(model, config, budget, self.c_args)
        

def select_model(worker, tmnt_config_space, total_iterations, result_logger, id_str, ns_port):
    tmnt_config = TMNTConfig(tmnt_config_space)
    worker.run(background=True)
    cs = tmnt_config.get_configspace() 
    config = cs.sample_configuration().get_dictionary()
    logging.info(config)
    bohb = BOHB(  configspace = cs,
                  run_id = id_str, nameserver='127.0.0.1', result_logger=result_logger, nameserver_port=ns_port,
                  min_budget=2, max_budget=worker.max_budget
           )
    res = bohb.run(n_iterations=total_iterations)
    bohb.shutdown(shutdown_workers=True)
    return res

def write_model(m, config, budget, args):
    if args.model_dir:
        pfile = os.path.join(args.model_dir, 'model.params')
        sp_file = os.path.join(args.model_dir, 'model.config')
        vocab_file = os.path.join(args.model_dir, 'vocab.json')
        m.save_parameters(pfile)
        ## if the embedding_size wasn't set explicitly (e.g. determined via pre-trained embedding), then set it here
        if m.vocabulary.embedding:
            config['embedding_size'] = len(m.vocabulary.embedding.idx_to_vec[0])
        config['training_epochs'] = int(budget)
        specs = json.dumps(config)
        with open(sp_file, 'w') as f:
            f.write(specs)
        with open(vocab_file, 'w') as f:
            f.write(m.vocabulary.to_json())


def get_worker(args, budget, id_str, ns_port):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second,i_dt.microsecond)
    logging_config(folder=train_out_dir, name='bow_ntm', level=logging.INFO)
    logging.info(args)
    seed_rng(args.seed)
    sp_vec_data = False
    ## if the vocab file and training files are available, use those
    if args.vocab_file and args.tr_vec_file:
        vpath = Path(args.vocab_file)
        tpath = Path(args.tr_vec_file)
        if vpath.is_file() and tpath.is_file():
            sp_vec_data = True
    if sp_vec_data:
        logging.info("Loading data via pre-computed vocabulary and sparse vector format document representation")
        vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words, tr_labels, tst_labels = \
            collect_sparse_data(args.tr_vec_file, args.vocab_file, args.tst_vec_file)
    else:
        raise Exception("Vocab file {} and/or training vector file {} do not exist".format(args.vocab_file, args.tr_vec_file))
    ctx = mx.cpu() if args.gpu is None or args.gpu == '' or int(args.gpu) < 0 else mx.gpu(int(args.gpu))
    ### XXX - NOTE: For smaller datasets, may make sense to convert sparse matrices to dense here up front
    worker = BowVAEWorker(args, vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words, tr_labels, tst_labels, ctx=ctx,
                              max_budget=budget,
                              nameserver='127.0.0.1', run_id=id_str, nameserver_port=ns_port)
    return worker, train_out_dir


def get_port():
    p = 9090
    while usedPort(p):
        p += 1
    return p

def model_select_bow_vae(args):
    dd = datetime.datetime.now()
    id_str = dd.strftime("%Y-%m-%d_%H-%M-%S")
    ns_port = get_port()
    worker, log_dir = get_worker(args, args.budget, id_str, ns_port)
    worker.search_mode = True
    result_logger = hpres.json_result_logger(directory=log_dir, overwrite=True)
    logging.info("Starting nameserver on port {}".format(ns_port))
    NS = hpns.NameServer(run_id=id_str, host='127.0.0.1', port=ns_port)
    NS.start()
    res = select_model(worker, args.config_space, args.iterations, result_logger, id_str, ns_port)
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    logging.info('Best found configuration:', id2config[incumbent]['config'])
    logging.info('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/32))
    # Here is how you get he incumbent (best configuration)
    # let's grab the run on the highest budget
    inc_runs = res.get_runs_by_id(incumbent)
    inc_run = inc_runs[-1]
    inc_loss = inc_run.loss
    inc_config = id2config[incumbent]['config']
    logging.info("Best configuration loss = {}".format(inc_loss))
    logging.info("Best configuration {}".format(inc_config))
    with open(os.path.join(log_dir, 'model_selection_results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    with open(os.path.join(log_dir, 'best.model.config'), 'w') as fp:
        inc_config['training_epochs'] = args.budget
        specs = json.dumps(inc_config)
        fp.write(specs)
    if args.model_dir:
        worker.retrain_best_config(inc_config, inc_run.budget)
    NS.shutdown()

def train_bow_vae(args):
    with open(args.config, 'r') as f:
        config = json.loads(f.read())
    dd = datetime.datetime.now()
    id_str = dd.strftime("%Y-%m-%d_%H-%M-%S")
    ns_port = get_port()
    worker, log_dir = get_worker(args, int(config['training_epochs']), id_str, ns_port)
    worker.retrain_best_config(config, int(config['training_epochs']))

