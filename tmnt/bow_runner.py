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


def get_wd_freqs(data_csr, max_sample_size=10000):
    sample_size = min(max_sample_size, data_csr.shape[0])
    data = data_csr[:sample_size].asnumpy() # only take first 10000 to estimate frequencies - but should select at RANDOM
    sums = np.sum(data, axis=0)
    return list(sums)


def train(args, vocabulary, data_train_csr, total_tr_words, data_test_csr=None, total_tst_words=0, train_labels=None, test_labels=None, ctx=mx.cpu()):
    wd_freqs = get_wd_freqs(data_train_csr)
    emb_size = vocabulary.embedding.idx_to_vec[0].size if vocabulary.embedding else args.embedding_size
    l1_coef = args.init_sparsity_pen if args.target_sparsity > 0.0 else 0.0
    seed_matrix = None
    if args.topic_seed_file:
        seed_matrix = get_seed_matrix_from_file(args.topic_seed_file, vocabulary)
    if args.use_labels_as_covars and train_labels is not None:
        n_covars = mx.nd.max(train_labels).asscalar() + 1
        train_labels = mx.nd.one_hot(train_labels, n_covars)
        test_labels = mx.nd.one_hot(test_labels, n_covars) if test_labels is not None else None
        model = \
            MetaDataBowNTM(n_covars,vocabulary, args.hidden_dim, args.n_latent, emb_size,
                           fixed_embedding=args.fixed_embedding, latent_distrib=args.latent_distribution,
                           init_l1=l1_coef, coherence_reg_penalty=args.coherence_regularizer_penalty,
                           batch_size=args.batch_size, wd_freqs=wd_freqs, ctx=ctx)
    else:
        model = \
            BowNTM(vocabulary, args.hidden_dim, args.n_latent, emb_size, fixed_embedding=args.fixed_embedding, latent_distrib=args.latent_distribution,
                   init_l1=l1_coef, coherence_reg_penalty=args.coherence_regularizer_penalty,
                   batch_size=args.batch_size, wd_freqs=wd_freqs, seed_mat=seed_matrix, ctx=ctx)

    train_iter = mx.io.NDArrayIter(data_train_csr, train_labels, args.batch_size, last_batch_handle='discard', shuffle=True)
    train_dataloader = DataIterLoader(train_iter)    
    if data_test_csr is not None:
        test_iter = mx.io.NDArrayIter(data_test_csr, test_labels, args.batch_size, last_batch_handle='discard', shuffle=False)
        test_dataloader = DataIterLoader(test_iter)        
    if (args.hybridize):
        model.hybridize(static_alloc=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        total_rec_loss = 0
        total_l1_pen = 0
        total_kl_loss = 0
        total_entropies_loss = 0
        total_coherence_loss = 0
        tr_size = 0
        for i, (data, labels) in enumerate(train_dataloader):
            tr_size += data.shape[0]
            if labels is None or labels.size == 0:
                labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
                labels = labels.as_in_context(ctx)
            data = data.as_in_context(ctx)
            with autograd.record():
                elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss, _ = model(data, labels) if args.use_labels_as_covars else model(data)
                elbo_mean = elbo.mean()
            elbo_mean.backward()
            trainer.step(data.shape[0]) ## step based on batch size
            total_kl_loss += kl_loss.sum().asscalar()
            total_l1_pen += l1_pen.sum().asscalar()
            total_rec_loss += rec_loss.sum().asscalar()
            if coherence_loss is not None:
                total_coherence_loss += coherence_loss
            if entropies is not None:
                total_entropies_loss += entropies.sum().asscalar()
            epoch_loss += elbo.sum().asscalar()
        logging.info("Epoch {}: Loss = {}, [ KL loss = {:8.4f} ] [ L1 loss = {:8.4f} ] [ Rec loss = {:8.4f}] [ Coherence loss = {:8.4f} ] [ Entropy losss = {:8.4f} ]".
                     format(epoch,
                            epoch_loss / tr_size,
                            total_kl_loss / tr_size,
                            total_l1_pen / tr_size,
                            total_rec_loss / tr_size,
                            total_coherence_loss / tr_size,                             
                            total_entropies_loss / tr_size))
        if args.target_sparsity > 0.0:            
            dec_weights = model.decoder.collect_params().get('weight').data().abs()
            ratio_small_weights = (dec_weights < args.sparsity_threshold).sum().asscalar() / dec_weights.size
            l1_coef = l1_coef * math.pow(2.0, args.target_sparsity - ratio_small_weights)
            logging.info("Setting L1 coeffficient to {} [sparsity ratio = {}]".format(l1_coef, ratio_small_weights))
            model.l1_pen_const.set_data(mx.nd.array([l1_coef]))
        if data_test_csr is not None and (epoch + 1) % args.eval_freq == 0:
            perplexity = evaluate(model, test_dataloader, total_tst_words, args, ctx)
            if args.trace_file:
                otype = 'a+' if epoch >= args.eval_freq else 'w+'
                with io.open(args.trace_file, otype) as fp:
                    if otype == 'w+':
                        fp.write("Epoch,PPL,NPMI\n")
                    npmi = compute_coherence(model, vocabulary, args.n_latent, 10, data_test_csr)
                    fp.write("{:3d},{:10.2f},{:8.4f}\n".format(epoch, perplexity, npmi))
    log_top_k_words_per_topic(model, vocabulary, args.n_latent, 10)
    compute_coherence(model, vocabulary, args.n_latent, 10, data_test_csr)
    if seed_matrix is not None:
        analyze_seed_matrix(model, seed_matrix)
    return model


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


def compute_coherence(model, vocab, num_topics, k, test_data):
    w = model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    num_topics = min(num_topics, sorted_ids.shape[-1])
    top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]
    npmi_eval = EvaluateNPMI(top_k_words_per_topic)
    npmi = npmi_eval.evaluate_csr_mat(test_data)
    logging.info("Test Coherence: {}".format(npmi))
    return npmi


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
                 data_test_csr, total_tst_words, train_labels=None, test_labels=None, ctx=mx.cpu(), max_budget=32, **kwargs):
        super().__init__(**kwargs)
        self.c_args = c_args
        self.total_tr_words = total_tr_words
        self.total_tst_words = total_tst_words
        self.vocabulary = vocabulary
        self.ctx = ctx
        self.data_test_csr = data_test_csr
        self.max_budget = max_budget
        
        self.wd_freqs = get_wd_freqs(data_train_csr)
        self.emb_size = vocabulary.embedding.idx_to_vec[0].size if vocabulary.embedding else c_args.embedding_size
        self.l1_coef = c_args.init_sparsity_pen if c_args.target_sparsity > 0.0 else 0.0
        self.seed_matrix = None
        if c_args.topic_seed_file:
            self.seed_matrix = get_seed_matrix_from_file(c_args.topic_seed_file, vocabulary)
        train_iter = mx.io.NDArrayIter(data_train_csr, train_labels, c_args.batch_size, last_batch_handle='discard', shuffle=True)
        self.train_dataloader = DataIterLoader(train_iter)    
        test_iter = mx.io.NDArrayIter(data_test_csr, test_labels, c_args.batch_size, last_batch_handle='discard', shuffle=False)
        self.test_dataloader = DataIterLoader(test_iter)        

    def compute(self, config, budget, working_directory, *args, **kwargs):

        lr = config['lr']
        latent_distrib = config['latent_distribution']
        optimizer = config['optimizer']
        n_latent = int(config['n_latent'])
        enc_hidden_dim = int(config['enc_hidden_dim'])
        
        if self.c_args.use_labels_as_covars and train_labels is not None:
            n_covars = mx.nd.max(train_labels).asscalar() + 1
            train_labels = mx.nd.one_hot(train_labels, n_covars)
            test_labels = mx.nd.one_hot(test_labels, n_covars) if test_labels is not None else None
            model = \
                MetaDataBowNTM(n_covars, self.vocabulary, enc_hidden_dim, n_latent, self.emb_size,
                               fixed_embedding=self.c_args.fixed_embedding, latent_distrib=self.c_args.latent_distribution,
                               init_l1=l1_coef, coherence_reg_penalty=self.c_args.coherence_regularizer_penalty,
                               batch_size=self.c_args.batch_size, wd_freqs=wd_freqs, ctx=ctx)
        else:
            model = \
                BowNTM(self.vocabulary, enc_hidden_dim, n_latent, self.emb_size,
                   fixed_embedding=self.c_args.fixed_embedding, latent_distrib=latent_distrib,
                   init_l1=self.l1_coef, coherence_reg_penalty=self.c_args.coherence_regularizer_penalty,
                   batch_size=self.c_args.batch_size, wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, ctx=self.ctx)

        trainer = gluon.Trainer(model.collect_params(), optimizer, {'learning_rate': lr})
        logging.info("Executing model evaluation with lr = {}, latent_distrib = {} and budget = {}".format(lr, latent_distrib, budget))
        for epoch in range(int(budget)):
            epoch_loss = 0
            total_rec_loss = 0
            total_l1_pen = 0
            total_kl_loss = 0
            total_entropies_loss = 0
            total_coherence_loss = 0
            tr_size = 0
            for i, (data, labels) in enumerate(self.train_dataloader):
                tr_size += data.shape[0]
                if labels is None or labels.size == 0:
                    labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
                    labels = labels.as_in_context(self.ctx)
                data = data.as_in_context(self.ctx)
                with autograd.record():
                    elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss, _ = model(data, labels) if self.c_args.use_labels_as_covars else model(data)
                    elbo_mean = elbo.mean()
                elbo_mean.backward()
                trainer.step(data.shape[0]) ## step based on batch size
                total_kl_loss += kl_loss.sum().asscalar()
                total_l1_pen += l1_pen.sum().asscalar()
                total_rec_loss += rec_loss.sum().asscalar()
                if coherence_loss is not None:
                    total_coherence_loss += coherence_loss
                if entropies is not None:
                    total_entropies_loss += entropies.sum().asscalar()
                epoch_loss += elbo.sum().asscalar()
            logging.info("Epoch {}: Loss = {}, [ KL loss = {:8.4f} ] [ L1 loss = {:8.4f} ] [ Rec loss = {:8.4f}] [ Coherence loss = {:8.4f} ] [ Entropy losss = {:8.4f} ]".
                     format(epoch,
                            epoch_loss / tr_size,
                            total_kl_loss / tr_size,
                            total_l1_pen / tr_size,
                            total_rec_loss / tr_size,
                            total_coherence_loss / tr_size,                             
                            total_entropies_loss / tr_size))
        perplexity = evaluate(model, self.test_dataloader, self.total_tst_words, self.c_args, self.ctx)
        npmi = compute_coherence(model, self.vocabulary, n_latent, 10, self.data_test_csr)
        logging.info("NPMI = {}".format(npmi))
        return ({
            'loss': 1.0 - npmi,
            'info': {
                'test_perplexity': perplexity,
                'train_loss': (epoch_loss / tr_size)
            }
        })
        

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        #lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-2, default_value='1e-3', log=True)

        #latent_distribution = CSH.CategoricalHyperparameter('latent_distribution', ['vmf', 'logistic_gaussian', 'gaussian'])

        #cs.add_hyperparameters([lr, latent_distribution])

        #kappa =  CSH.UniformIntegerHyperparameter('kappa', lower=1.0, upper=200.0, default_value=100.0)
        #cs.add_hyperparameters([kappa])
        
        #kappa_cond = CS.EqualsCondition(kappa, latent_distribution, 'vmf')
        #cs.add_condition(kappa_cond)

        #dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.2, log=False)
        
        #cs.add_hyperparameters([dropout_rate])
        
        return cs


def select_model(worker, tmnt_config_file):
    tmnt_config = TMNTConfig(tmnt_config_file)
    worker.run(background=True)
    cs = tmnt_config.get_configspace()  ##worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    bohb = BOHB(  configspace = cs,
              run_id = '0', nameserver='127.0.0.1',
              min_budget=2, max_budget=32
           )
    res = bohb.run(n_iterations=8)
    bohb.shutdown(shutdown_workers=True)
    return res


def train_bow_vae(args):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
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
        logging.info("Loading and pre-processing text data found in {}".format(args.train_dir))
        tr_dataset = BowDataSet(args.train_dir, args.file_pat)    
        tr_csr_mat, vocab, total_tr_words = collect_stream_as_sparse_matrix(tr_dataset, max_vocab_size=args.max_vocab_size)
        tr_labels = None
        tst_labels = None
        if args.vocab_file and args.tr_vec_file:
            export_sparse_matrix(tr_csr_mat, args.tr_vec_file)
            export_vocab(vocab, args.vocab_file)
        if args.test_dir:
            tst_dataset = BowDataSet(args.test_dir, args.file_pat)
            tst_csr_mat, _, total_tst_words = collect_stream_as_sparse_matrix(tst_dataset, pre_vocab=vocab)
            if args.vocab_file and args.tst_vec_file:
                export_sparse_matrix(tst_csr_mat, args.tst_vec_file)
    ctx = mx.cpu() if args.gpu is None or args.gpu == '' or int(args.gpu) < 0 else mx.gpu(int(args.gpu))
    if args.embedding_source:
        glove_twitter = nlp.embedding.create('glove', source=args.embedding_source)
        vocab.set_embedding(glove_twitter)
        emb_size = len(vocab.embedding.idx_to_vec[0])
        num_oov = 0
        for word in vocab.embedding._idx_to_token:
            if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                logging.info("Term {} is OOV".format(word))
                num_oov += 1
                vocab.embedding[word] = mx.nd.random.normal(0, 1.0, emb_size)
        logging.info(">> {} Words did not appear in embedding source {}".format(num_oov, args.embedding_source))
    ### XXX - NOTE: For smaller datasets, may make sense to convert sparse matrices to dense here up front
    if args.model_select:
        worker = BowVAEWorker(args, vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words, tr_labels, tst_labels, ctx=ctx,
                              max_budget=args.epochs,
                              nameserver='127.0.0.1', run_id='0')
        NS = hpns.NameServer(run_id='0', host='127.0.0.1', port=None)
        NS.start()
        res = select_model(worker, args.config_file)
        NS.shutdown()
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/32))
        # Here is how you get he incumbent (best configuration)
        incumbent = res.get_incumbent_id()
        # let's grab the run on the highest budget
        inc_runs = res.get_runs_by_id(incumbent)
        inc_run = inc_runs[-1]
        inc_loss = inc_run.loss
        inc_config = id2conf[inc_id]['config']
        inc_test_loss = inc_run.info['test accuracy']
        with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
            pickle.dump(res, fh)
    else:
        m = train(args, vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words, tr_labels, tst_labels, ctx=ctx)
        if args.model_dir:
            pfile = os.path.join(args.model_dir, 'model.params')
            sp_file = os.path.join(args.model_dir, 'model.specs')
            vocab_file = os.path.join(args.model_dir, 'vocab.json')
            m.save_parameters(pfile)
            sp = {}
            sp['enc_dim'] = args.hidden_dim
            sp['n_latent'] = args.n_latent
            sp['latent_distribution'] = args.latent_distribution
            sp['emb_size'] = vocab.embedding.idx_to_vec[0].size if vocab.embedding else args.embedding_size
            specs = json.dumps(sp)
            with open(sp_file, 'w') as f:
                f.write(specs)
            with open(vocab_file, 'w') as f:
                f.write(vocab.to_json())
