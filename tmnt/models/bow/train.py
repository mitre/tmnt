# coding: utf-8
"""
Copyright (c) 2019-2020 The MITRE Corporation.
"""

import math
import logging
import datetime
import time
import io
import os
import json
import mxnet as mx
import numpy as np
import pickle
import copy
import socket
import statistics
import random
import pandas as pd

from mxnet import autograd
from mxnet import gluon
import gluonnlp as nlp
from pathlib import Path

from tmnt.models.bow.bow_doc_loader import DataIterLoader, collect_sparse_test, collect_sparse_data, load_vocab
from tmnt.models.bow.bow_vae import BowVAE, MetaBowVAE
from tmnt.models.bow.topic_seeds import get_seed_matrix_from_file
from tmnt.models.bow.sensitivity_analysis import get_encoder_jacobians_at_data_nocovar

from tmnt.utils.log_utils import logging_config
from tmnt.utils.mat_utils import export_sparse_matrix, export_vocab
from tmnt.utils.random import seed_rng
from tmnt.coherence.npmi import EvaluateNPMI
from tmnt.modsel.configuration import TMNTConfig

import autogluon as ag
from autogluon.scheduler.reporter import FakeReporter

from tabulate import tabulate

__all__ = ['model_select_bow_vae', 'train_bow_vae']

MAX_DESIGN_MATRIX = 250000000 

def get_wd_freqs(data_csr, max_sample_size=1000000):
    sample_size = min(max_sample_size, data_csr.shape[0])
    data = data_csr[:sample_size] 
    sums = mx.nd.sum(data, axis=0)
    return sums


def evaluate(model, epoch, data_loader, last_batch_size, num_test_batches, total_words, args, ctx=mx.cpu()):
    total_rec_loss = 0
    total_kl_loss  = 0
    batch_size = 0
    for i, (data,labels) in enumerate(data_loader):
        if labels is None:            
            labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
        data = data.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        _, kl_loss, rec_loss, _, _, _, _, log_out = model(data, labels) if args.use_labels_as_covars else model(data)
        ## We explicitly keep track of the last batch size        
        ## The following lets us use a "rollover" for handling the last batch,
        ## enabling the symbolic computation graph (via hybridize)
        if i == num_test_batches - 1:
            total_rec_loss += rec_loss[:last_batch_size].sum().asscalar()
            total_kl_loss  += kl_loss[:last_batch_size].sum().asscalar()
        else:
            total_rec_loss += rec_loss.sum().asscalar()
            total_kl_loss += kl_loss.sum().asscalar()
    if ((total_rec_loss + total_kl_loss) / total_words) < 709.0:
        perplexity = math.exp((total_rec_loss + total_kl_loss) / total_words)
    else:
        perplexity = 1e300
    logging.info("TEST/VALIDATION (Epoch {}) Perplexity = {} [ Rec Loss = {} + KL loss = {} / Total test words = {}]".
                 format(epoch, perplexity, total_rec_loss, total_kl_loss, total_words))
    return perplexity



def compute_coherence(model, k, test_data, log_terms=False, covariate_interactions=False,
                      test_dataloader=None, ctx=mx.cpu()):
    if covariate_interactions:
        logging.debug("Rendering interactions not supported yet")
    num_topics = model.n_latent

    if test_dataloader is not None:
        ## in this case compute coherence using encoder Jacobian
        js = get_encoder_jacobians_at_data_nocovar(model, test_dataloader, int(1000 / num_topics), 100000, ctx)
        sorted_j = (-js).argsort(axis=1)
        sorted_topk = sorted_j[:, :k]
        enc_top_k_words_per_topic = [ [int(i) for i in list(sorted_topk[t, :]) ] for t in range(num_topics)]
        enc_npmi_eval = EvaluateNPMI(enc_top_k_words_per_topic)
        enc_npmi = enc_npmi_eval.evaluate_csr_mat(test_data)
    else:
        enc_npmi = None
        
    sorted_ids = model.get_top_k_terms(k)
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
            logging.debug("Topic {}: {}".format(i, top_k_tokens[i]))
    return npmi, enc_npmi, redundancy


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
        logging.debug("Topic {}: {}".format(str(t), term_str))


def x_get_mxnet_visible_gpus():
    import mxnet as mx
    gpu_count = 0
    while True:
        try:
            arr = mx.np.array(1.0, ctx=mx.gpu(gpu_count))
            arr.asnumpy()
            gpu_count += 1
        except Exception:
            break
    return [mx.gpu(i) for i in range(gpu_count)]


def get_mxnet_visible_gpus():
    ln = 0
    t = datetime.datetime.now()
    while ln < 1 and ((datetime.datetime.now() - t).seconds < 30):
        time.sleep(1)
        gpus = x_get_mxnet_visible_gpus()
        ln = len(gpus)
    if ln > 0:
        return gpus
    else:
        raise Exception("Unable to get a gpu after 30 tries")
        

class BowVAETrainer():
    def __init__(self, model_out_dir, c_args, vocabulary, data_train_csr, total_tr_words,
                 data_test_csr, total_tst_words, train_labels=None, test_labels=None, label_map=None, use_gpu=False):
        self.model_out_dir = model_out_dir
        self.c_args = c_args
        self.use_gpu = use_gpu
        self.total_tr_words = total_tr_words
        self.total_tst_words = total_tst_words
        self.vocabulary = vocabulary
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.data_train_csr   = data_train_csr
        self.data_test_csr    = data_test_csr
        self.data_heldout_csr = None
        self.label_map = label_map
        self.wd_freqs = get_wd_freqs(data_train_csr)
        self.vocab_cache = {}
        self.validate_each_epoch = True
        self.seed_matrix = None
        self.vocab_cache = {}
        if c_args.topic_seed_file:
            self.seed_matrix = get_seed_matrix_from_file(c_args.topic_seed_file, vocabulary, ctx)


    def pre_build_vocab_cache(self, possible_vocabs):
        """
        Preload and cache all vocabularies for faster model selection
        """
        raise NotImplementedError()

    def _initialize_vocabulary(self, embedding_source, set_vocab=True):
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
        emb_size: Size of embedding (based on pre-trained embedding, random returns -1 for size to be set later)
        """
        vocab = self.vocabulary
        if embedding_source != 'random':
            if self.vocab_cache.get(embedding_source):
                pt_embedding = copy.deepcopy(self.vocab_cache[embedding_source])
            else:
                e_type, e_name = tuple(embedding_source.split(':'))
                if e_type == 'file':
                    if not os.path.exists(e_name):
                        raise Exception("Embedding file not found: {}".format(e_name))
                    pt_embedding = nlp.embedding.TokenEmbedding.from_file(e_name)
                else:
                    pt_embedding = nlp.embedding.create(e_type, source=e_name)
                #vocab = copy.deepcopy(self.vocabulary) ## create a copy of the vocab to attach the vocab to
                self.vocab_cache[embedding_source] = copy.deepcopy(pt_embedding) ## cache another copy 
            emb_size = len(pt_embedding.idx_to_vec[0])
            if set_vocab:
                vocab.set_embedding(pt_embedding)
                num_oov = 0
                for word in vocab.embedding._idx_to_token:
                    if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                        logging.debug("Term {} is OOV".format(word))
                        num_oov += 1
                        vocab.embedding[word] = mx.nd.random.normal(0, 0.1, emb_size)
                logging.debug(">> {} Words did not appear in embedding source {}".format(num_oov, embedding_source))
        else:
            vocab.set_embedding(None) ## unset embedding
            emb_size = -1
        return vocab, emb_size

    def pre_cache_vocabularies(self, sources):
        for s in sources:
            self._initialize_vocabulary(s, set_vocab=False)

    def set_validate_each_epoch(self, v):
        self.validate_each_epoch = v

    def set_heldout_data_as_test(self):
        """Load in the heldout test data for final model evaluation
        """
        tst_mat, total_tst_words, tst_labels = collect_sparse_test(self.c_args.tst_vec_file, self.vocabulary,
                                                                   scalar_labels=self.c_args.scalar_covars,
                                                                   encoding=self.c_args.str_encoding)
        self.data_test_csr = tst_mat
        self.test_labels   = tst_labels
        self.total_tst_words = total_tst_words
        

    def _get_vae_model(self, config, reporter, ctx):
        """Take a model configuration - specified by a config file or as determined by model selection and 
        return a VAE topic model ready for training.

        Parameters
        ----------
        config: an autogluon configuration/argument object 
        """
        
        lr = config.lr
        latent_distrib = config.latent_distribution
        optimizer = config.optimizer
        n_latent = int(config.n_latent)
        enc_hidden_dim = int(config.enc_hidden_dim)
        coherence_reg_penalty = float(config.coherence_loss_wt)
        redundancy_reg_penalty = float(config.redundancy_loss_wt)
        batch_size = int(config.batch_size)

        embedding_source = config.embedding.source
        fixed_embedding  = config.embedding.get('fixed') == True
        covar_net_layers = config.covar_net_layers
        n_encoding_layers = config.num_enc_layers
        enc_dr = config.enc_dr
        epochs = int(config.epochs)

        ldist_def = config.latent_distribution
        kappa = 0.0
        alpha = 1.0
        latent_distrib = ldist_def.dist_type
        if latent_distrib == 'vmf':
            kappa = ldist_def.kappa
        elif latent_distrib == 'logistic_gaussian':
            alpha = ldist_def.alpha

        vocab, emb_size = self._initialize_vocabulary(embedding_source)
        if emb_size < 0 and 'size' in config.embedding:
            emb_size = config.embedding.size

        if self.c_args.use_labels_as_covars and self.train_labels is not None:
            n_covars = len(self.label_map) if self.label_map else 1
            model = \
                MetaBowVAE(vocab, coherence_coefficient=8.0, reporter=reporter, label_map=self.label_map,
                           covar_net_layers=1, ctx=ctx, lr=lr, latent_distribution=latent_distrib, optimizer=optimizer,
                           n_latent=n_latent, kappa=kappa, alpha=alpha, enc_hidden_dim=enc_hidden_dim,
                           coherence_reg_penalty=coherence_reg_penalty,
                           redundancy_reg_penalty=redundancy_reg_penalty, batch_size=batch_size, 
                           embedding_source=embedding_source, embedding_size=emb_size, fixed_embedding=fixed_embedding,
                           num_enc_layers=n_encoding_layers, enc_dr=enc_dr, seed_matrix=self.seed_matrix, hybridize=False,
                           epochs=epochs)
        else:
            model = \
                BowVAE(vocab, coherence_coefficient=8.0, reporter=reporter, ctx=ctx, lr=lr, latent_distribution=latent_distrib, optimizer=optimizer,
                       n_latent=n_latent, kappa=kappa, alpha=alpha, enc_hidden_dim=enc_hidden_dim,
                       coherence_reg_penalty=coherence_reg_penalty,
                       redundancy_reg_penalty=redundancy_reg_penalty, batch_size=batch_size, 
                       embedding_source=embedding_source, embedding_size=emb_size, fixed_embedding=fixed_embedding,
                       num_enc_layers=n_encoding_layers, enc_dr=enc_dr, seed_matrix=self.seed_matrix, hybridize=False,
                       epochs=epochs)
            model.validate_each_epoch = self.validate_each_epoch
        return model
    

    def train_model(self, config, reporter):
        """Main training function which takes a single model configuration and a budget (i.e. number of epochs) and
        fits the model to the training data.
        
        Parameters
        ----------
        config: `Configuration` object within the specified `ConfigSpace`
        reporter: Reporter callback for model selection

        Returns
        -------
        model: VAE model with trained parameters
        obj: scaled objective
        npmi: coherence on validation set
        perplexity: perplexity score on validation data
        redundancy: topic model redundancy of top 5 terms for each topic
        """
        logging.debug("Evaluating with Config: {}".format(config))
        ctx_list = get_mxnet_visible_gpus() if self.use_gpu else [mx.cpu()]
        ctx = ctx_list[0]
        vae_model = self._get_vae_model(config, reporter, ctx)
        obj, npmi, perplexity, redundancy = vae_model.fit_with_validation(self.data_train_csr, self.train_labels, self.data_test_csr, self.test_labels)
        return vae_model.model, obj, npmi, perplexity, redundancy



def write_model(m, model_dir, config):
    if model_dir:
        pfile = os.path.join(model_dir, 'model.params')
        sp_file = os.path.join(model_dir, 'model.config')
        vocab_file = os.path.join(model_dir, 'vocab.json')
        logging.info("Model parameters, configuration and vocabulary written to {}".format(model_dir))
        m.save_parameters(pfile)
        ## additional derived information from auto-searched configuration
        ## helpful to have for runtime use of model (e.g. embedding size)
        derived_info = {}
        derived_info['embedding_size'] = m.embedding_size
        config['derived_info'] = derived_info
        if 'num_enc_layers' not in config.keys():
            config['num_enc_layers'] = m.num_enc_layers
            config['n_covars'] = int(m.n_covars)
            config['l_map'] = m.label_map
            config['covar_net_layers'] = m.covar_net_layers
        specs = json.dumps(config, sort_keys=True, indent=4)
        with open(sp_file, 'w') as f:
            f.write(specs)
        with open(vocab_file, 'w') as f:
            f.write(m.vocabulary.to_json())


def get_trainer(c_args):
    i_dt = datetime.datetime.now()
    train_out_dir = \
        os.path.join(c_args.save_dir,
                     "train_{}_{}_{}_{}_{}_{}_{}".format(i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second,i_dt.microsecond))
    ll = c_args.log_level
    log_level = logging.INFO
    if ll.lower() == 'info':
        log_level = logging.INFO
    elif ll.lower() == 'debug':
        log_level = logging.DEBUG
    elif ll.lower() == 'error':
        log_level = logging.ERROR
    elif ll.lower() == 'warning':
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    logging_config(folder=train_out_dir, name='tmnt', level=log_level, console_level=log_level)
    logging.info(c_args)
    seed_rng(c_args.seed)
    if c_args.vocab_file and c_args.tr_vec_file:
        vpath = Path(c_args.vocab_file)
        tpath = Path(c_args.tr_vec_file)
        if not (vpath.is_file() and tpath.is_file()):
            raise Exception("Vocab file {} and/or training vector file {} do not exist".format(c_args.vocab_file, c_args.tr_vec_file))
    logging.info("Loading data via pre-computed vocabulary and sparse vector format document representation")
    vocab, tr_csr_mat, total_tr_words, tr_labels, label_map = \
        collect_sparse_data(c_args.tr_vec_file, c_args.vocab_file, scalar_labels=c_args.scalar_covars, encoding=c_args.str_encoding)
    if c_args.val_vec_file:
        tst_csr_mat, total_tst_words, tst_labels = \
            collect_sparse_test(c_args.val_vec_file, vocab, scalar_labels=c_args.scalar_covars, encoding=c_args.str_encoding)
    else:
        tst_csr_mat, total_tst_words, tst_labels = None, None, None
    ctx = mx.cpu() if not c_args.use_gpu else mx.gpu(0)
    model_out_dir = c_args.model_dir if c_args.model_dir else os.path.join(train_out_dir, 'MODEL')
    if not os.path.exists(model_out_dir):
        os.mkdir(model_out_dir)
    if c_args.use_labels_as_covars and tr_labels is not None:
        if label_map is not None:
            n_covars = len(label_map)
            tr_labels = mx.nd.one_hot(tr_labels, n_covars)
            tst_labels = mx.nd.one_hot(tst_labels, n_covars) if tst_labels is not None else None
        else:
            tr_labels = mx.nd.expand_dims(tr_labels, 1)
            tst_labels = mx.nd.expand_dims(tst_labels, 1) if tst_labels is not None else None
    trainer = BowVAETrainer(model_out_dir, c_args, vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words, tr_labels, tst_labels,
                           label_map, use_gpu=c_args.use_gpu)
    return trainer, train_out_dir


def process_training_history(task_dicts, start_timestamp):
    task_dfs = []
    for task_id in task_dicts:
        task_df = pd.DataFrame(task_dicts[task_id])
        task_df = task_df.assign(task_id=task_id,
                                 coherence=task_df["coherence"],
                                 perplexity=task_df["perplexity"],
                                 redundancy=task_df["redundancy"],
                                 runtime=task_df["time_step"] - start_timestamp,
                                 objective=task_df["objective"],
                                 target_epoch=task_df["epoch"].iloc[-1])
        task_dfs.append(task_df)

    result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)
    # re-order by runtime
    result = result.sort_values(by="runtime")
    # calculate incumbent best -- the cumulative minimum of the error.
    result = result.assign(best=result["objective"].cummax())
    return result


def select_model(trainer, c_args):
    """
    Top level call to model selection. 
    """
    tmnt_config_space = c_args.config_space
    total_iterations = c_args.iterations
    cpus_per_task = c_args.cpus_per_task
    searcher = c_args.searcher
    brackets = c_args.brackets
    tmnt_config = TMNTConfig(tmnt_config_space).get_configspace()
    ##
    if not (c_args.scheduler == 'hyperband'):
        trainer.set_validate_each_epoch(False)

    ## pre-cache vocabularies
    sources = [ e['source'] for e in tmnt_config.get('embedding').data if e['source'] != 'random' ]
    logging.info('>> Pre-caching pre-trained embeddings/vocabularies: {}'.format(sources))
    trainer.pre_cache_vocabularies(sources)
    
    @ag.args(**tmnt_config)
    def exec_train_fn(args, reporter):
        return trainer.train_model(args, reporter)

    search_options = {
        'num_init_random': 2,
        'debug_log': True}

    num_gpus = 1 if c_args.use_gpu else 0
    if c_args.scheduler == 'hyperband':
        hpb_scheduler = ag.scheduler.HyperbandScheduler(
            exec_train_fn,
            resource={'num_cpus': cpus_per_task, 'num_gpus': num_gpus},
            searcher=searcher,
            search_options=search_options,
            num_trials=total_iterations,             #time_out=120,
            time_attr='epoch',
            reward_attr='objective',
            type='stopping',
            grace_period=1,
            reduction_factor=3,
            brackets=brackets)
    else:
        hpb_scheduler = ag.scheduler.FIFOScheduler(
            exec_train_fn,
            resource={'num_cpus': cpus_per_task, 'num_gpus': num_gpus},
            searcher=searcher,
            search_options=search_options,
            num_trials=total_iterations,             #time_out=120
            time_attr='epoch',
            reward_attr='objective',
            )
    hpb_scheduler.run()
    hpb_scheduler.join_jobs()
    return hpb_scheduler

def train_with_single_config(c_args, trainer, best_config):
    rng_seed = c_args.seed
    best_obj = -1000000000.0
    best_model = None
    if c_args.val_vec_file:
        trainer.set_validate_each_epoch(False)
        if c_args.tst_vec_file:
            trainer.set_heldout_data_as_test()        
        logging.info("Training with config: {}".format(best_config))
        npmis, perplexities, redundancies, objectives = [],[],[],[]
        ntimes = int(c_args.num_final_evals)
        for i in range(ntimes):
            seed_rng(rng_seed) # update RNG
            rng_seed += 1
            model, obj, npmi, perplexity, redundancy = trainer.train_model(best_config, FakeReporter())
            npmis.append(npmi)
            perplexities.append(perplexity)
            redundancies.append(redundancy)
            objectives.append(obj)
            if obj > best_obj:
                best_obj = obj
                best_model = model
        test_type = "HELDOUT" if c_args.tst_vec_file else "VALIDATION"
        if ntimes > 1:
            logging.info("Final {} NPMI         ==> Mean: {}, StdDev: {}"
                         .format(test_type, statistics.mean(npmis), statistics.stdev(npmis)))
            logging.info("Final {} Perplexity   ==> Mean: {}, StdDev: {}"
                         .format(test_type, statistics.mean(perplexities), statistics.stdev(perplexities)))
            logging.info("Final {} Redundancy   ==> Mean: {}, StdDev: {}"
                         .format(test_type, statistics.mean(redundancies), statistics.stdev(redundancies)))
            logging.info("Final {} Objective    ==> Mean: {}, StdDev: {}"
                         .format(test_type, statistics.mean(objectives), statistics.stdev(objectives)))
        else:
            logging.info("Final {} NPMI         ==> {}".format(test_type, npmis[0]))
            logging.info("Final {} Perplexity   ==> {}".format(test_type, perplexities[0]))
            logging.info("Final {} Redundancy   ==> {}".format(test_type, redundancies[0]))
            logging.info("Final {} Objective    ==> {}".format(test_type, objectives[0]))            
        return best_model, best_obj
    else:
        model, obj, _, _, _ = trainer.train_model(best_config, FakeReporter())
        return model, obj
        
def model_select_bow_vae(c_args):
    dd = datetime.datetime.now()
    trainer, log_dir = get_trainer(c_args)
    scheduler = select_model(trainer, c_args)
    best_config_spec = scheduler.get_best_config()
    args_dict = ag.space.Dict(**scheduler.train_fn.args)
    best_config = args_dict.sample(**best_config_spec)
    logging.info("Best configuration = {}".format(best_config))
    logging.info("Best configuration objective = {}".format(scheduler.get_best_reward()))
    best_config_dict = ag.space.Dict(**best_config)
    logging.info("******************************* RETRAINING WITH BEST CONFIGURATION **************************")
    model, obj = train_with_single_config(c_args, trainer, best_config_dict)
    logging.info("Objective with final retrained model: {}".format(obj))
    write_model(model, trainer.model_out_dir, best_config)
    with open(os.path.join(log_dir, 'best.model.config'), 'w') as fp:
        specs = json.dumps(best_config)
        fp.write(specs)
    dd_finish = datetime.datetime.now()
    logging.info("Model selection run FINISHED. Time: {}".format(dd_finish - dd))
    results_df = process_training_history(
                scheduler.training_history.copy(),
                start_timestamp=scheduler._start_time)
    logging.info("Printing hyperparameter results")
    out_html = os.path.join(log_dir, 'selection.html')
    results_df.to_html(out_html)
    out_pretty = os.path.join(log_dir, 'selection.table.txt')
    with io.open(out_pretty, 'w') as fp:
        fp.write(tabulate(results_df, headers='keys', tablefmt='pqsl'))
    #logging.info("Providing model selection plot")
    #scheduler.get_training_curves(plot=True)
    scheduler.shutdown()

def train_bow_vae(args):
    try:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    except:
        logging.error("File passed to --config, {}, does not appear to be a valid .json configuration instance".format(args.config))
        raise Exception("Invalid Json Configuration File")
    dd = datetime.datetime.now()
    trainer, log_dir = get_trainer(args)
    config = ag.space.Dict(**config_dict)
    model, obj = train_with_single_config(args, trainer, config)
    write_model(model, trainer.model_out_dir, config_dict)
    dd_finish = datetime.datetime.now()
    logging.info("Model training FINISHED. Time: {}".format(dd_finish - dd))


