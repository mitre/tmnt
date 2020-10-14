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

from tmnt.models.base.base_trainer import BaseTrainer
from tmnt.models.base.base_selector import BaseSelector
from tmnt.models.bow.bow_doc_loader import DataIterLoader, load_vocab, file_to_data
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

__all__ = ['model_select_bow_vae', 'train_bow_vae']

MAX_DESIGN_MATRIX = 250000000 

        

class BowVAETrainer(BaseTrainer):
    def __init__(self, model_out_dir, c_args, vocabulary, wd_freqs, train_data, 
                 test_data, total_tst_words, train_labels=None, test_labels=None, label_map=None, use_gpu=False, val_each_epoch=True, rng_seed=1234):
        super().__init__(vocabulary, train_data, test_data, train_labels, test_labels, rng_seed)
        self.model_out_dir = model_out_dir
        self.c_args = c_args
        self.use_gpu = use_gpu
        self.total_tst_words = total_tst_words
        self.label_map = label_map
        self.wd_freqs = wd_freqs
        self.validate_each_epoch = val_each_epoch
        self.seed_matrix = None
        if c_args.topic_seed_file:
            self.seed_matrix = get_seed_matrix_from_file(c_args.topic_seed_file, vocabulary, ctx)


    def pre_cache_vocabularies(self, sources):
        for s in sources:
            self._initialize_vocabulary(s, set_vocab=False)

    def set_heldout_data_as_test(self):
        """Load in the heldout test data for final model evaluation
        """
        tst_mat, tst_labels, _, total_tst_words  = file_to_data(self.c_args.tst_vec_file, self.vocabulary)
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

        if self.c_args.use_labels_as_covars:
            #n_covars = len(self.label_map) if self.label_map else 1
            n_covars = -1
            model = \
                MetaBowVAE(vocab, coherence_coefficient=8.0, reporter=reporter, num_val_words=self.total_tst_words, wd_freqs=self.wd_freqs,
                           label_map=self.label_map,
                           covar_net_layers=1, ctx=ctx, lr=lr, latent_distribution=latent_distrib, optimizer=optimizer,
                           n_latent=n_latent, kappa=kappa, alpha=alpha, enc_hidden_dim=enc_hidden_dim,
                           coherence_reg_penalty=coherence_reg_penalty,
                           redundancy_reg_penalty=redundancy_reg_penalty, batch_size=batch_size, 
                           embedding_source=embedding_source, embedding_size=emb_size, fixed_embedding=fixed_embedding,
                           num_enc_layers=n_encoding_layers, enc_dr=enc_dr, seed_matrix=self.seed_matrix, hybridize=False,
                           epochs=epochs, log_method='log')
        else:
            model = \
                BowVAE(vocab, coherence_coefficient=8.0, reporter=reporter, num_val_words=self.total_tst_words, wd_freqs=self.wd_freqs,
                       ctx=ctx, lr=lr, latent_distribution=latent_distrib, optimizer=optimizer,
                       n_latent=n_latent, kappa=kappa, alpha=alpha, enc_hidden_dim=enc_hidden_dim,
                       coherence_reg_penalty=coherence_reg_penalty,
                       redundancy_reg_penalty=redundancy_reg_penalty, batch_size=batch_size, 
                       embedding_source=embedding_source, embedding_size=emb_size, fixed_embedding=fixed_embedding,
                       num_enc_layers=n_encoding_layers, enc_dr=enc_dr, seed_matrix=self.seed_matrix, hybridize=False,
                       epochs=epochs, log_method='log')
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
        ctx_list = self._get_mxnet_visible_gpus() if self.use_gpu else [mx.cpu()]
        ctx = ctx_list[0]
        vae_model = self._get_vae_model(config, reporter, ctx)
        obj, npmi, perplexity, redundancy = vae_model.fit_with_validation(self.train_data, self.train_labels, self.test_data, self.test_labels)
        return vae_model.model, obj, npmi, perplexity, redundancy

    def write_model(self, m, config):
        model_dir = self.model_out_dir
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
        else:
            raise Exception("Model write failed, output directory not provided")


def get_trainer(c_args, val_each_epoch=True):
    i_dt = datetime.datetime.now()
    train_out_dir = \
        os.path.join(c_args.save_dir,
                     "train_{}_{}_{}_{}_{}_{}_{}"
                     .format(i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second,i_dt.microsecond))
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
            raise Exception("Vocab file {} and/or training vector file {} do not exist"
                            .format(c_args.vocab_file, c_args.tr_vec_file))
    logging.info("Loading data via pre-computed vocabulary and sparse vector format document representation")
    vocab = load_vocab(c_args.vocab_file, encoding=c_args.str_encoding)
    voc_size = len(vocab)
    X, y, wd_freqs, _ = file_to_data(c_args.tr_vec_file, voc_size)
    total_test_wds = 0    
    if c_args.val_vec_file:
        val_X, val_y, _, total_test_wds = file_to_data(c_args.val_vec_file, voc_size)
    else:
        val_X, val_y, total_test_wds = None, None, 0
    ctx = mx.cpu() if not c_args.use_gpu else mx.gpu(0)
    model_out_dir = c_args.model_dir if c_args.model_dir else os.path.join(train_out_dir, 'MODEL')
    if not os.path.exists(model_out_dir):
        os.mkdir(model_out_dir)
    trainer = BowVAETrainer(model_out_dir, c_args, vocab, wd_freqs, X, val_X, total_test_wds,
                            train_labels = y, test_labels = val_y,
                            label_map=None, use_gpu=c_args.use_gpu, val_each_epoch=val_each_epoch)
    return trainer, train_out_dir


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

    ## pre-cache vocabularies before model selection (to avoid reloading for each model fit)
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


def model_select_bow_vae(c_args):
    tmnt_config = TMNTConfig(c_args.config_space).get_configspace()
    trainer, log_dir = get_trainer(c_args, val_each_epoch = (not (c_args.searcher == 'random')))
    selector = BaseSelector(tmnt_config,
                            c_args.iterations,
                            c_args.searcher,
                            c_args.scheduler,
                            c_args.brackets,
                            c_args.cpus_per_task,
                            c_args.use_gpu,
                            c_args.num_final_evals,
                            c_args.seed,
                            log_dir)
    sources = [ e['source'] for e in tmnt_config.get('embedding').data if e['source'] != 'random' ]
    logging.info('>> Pre-caching pre-trained embeddings/vocabularies: {}'.format(sources))
    trainer.pre_cache_vocabularies(sources)
    selector.select_model(trainer)
    

def train_bow_vae(args):
    try:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    except:
        logging.error("File passed to --config, {}, does not appear to be a valid .json configuration instance".format(args.config))
        raise Exception("Invalid Json Configuration File")
    dd = datetime.datetime.now()
    trainer, log_dir = get_trainer(args, val_each_epoch=args.eval_each_epoch)
    config = ag.space.Dict(**config_dict)
    model, obj = trainer.train_with_single_config(config, args.num_final_evals)
    trainer.write_model(model, config_dict)
    dd_finish = datetime.datetime.now()
    logging.info("Model training FINISHED. Time: {}".format(dd_finish - dd))


