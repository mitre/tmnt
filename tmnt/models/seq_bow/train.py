# codeing: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import math
import os
import numpy as np
import logging
import json
import datetime
import io
import gluonnlp as nlp
import string
import re
import mxnet as mx
import mxnet.ndarray as F
import autogluon as ag
from mxnet import gluon
from mxnet.gluon import nn
from tmnt.coherence.npmi import EvaluateNPMI
from tmnt.models.base.base_trainer import BaseTrainer
from tmnt.models.bow.bow_doc_loader import load_vocab
from tmnt.models.seq_bow.seq_bow_ved import SeqBowVED
from tmnt.models.seq_bow.models import TransformerBowVED
from tmnt.utils.log_utils import logging_config
from tmnt.models.seq_bow.sb_data_loader import load_dataset_basic_seq_bow, load_dataset_bert
from tmnt.modsel.configuration import TMNTConfig

__all__ = ['train_main']

def get_wd_freqs(data_csr, max_sample_size=1000000):
    sample_size = min(max_sample_size, data_csr.shape[0])
    data = data_csr[:sample_size] 
    sums = mx.nd.sum(data, axis=0)
    return sums


class SeqBowVEDTrainer(BaseTrainer):
    def __init__(self, model_out_dir, sent_size, vocabulary, wd_freqs, num_val_words, warmup_ratio, train_data, 
                 test_data, train_labels=None, test_labels=None, use_gpu=False, log_interval=10, rng_seed=1234):
        super().__init__(vocabulary, train_data, test_data, train_labels, test_labels, rng_seed)
        self.model_out_dir = model_out_dir
        self.use_gpu = use_gpu
        self.wd_freqs = wd_freqs
        self.seed_matrix = None
        self.sent_size = sent_size
        self.kld_wt = 1.0
        self.warmup_ratio = warmup_ratio
        self.num_val_words = num_val_words
        self.log_interval = log_interval

    def _get_ved_model(self, config, reporter, ctx):
        gen_lr = config.gen_lr
        dec_lr = config.dec_lr
        min_lr = config.min_lr
        optimizer = config.optimizer
        n_latent = int(config.n_latent)
        batch_size = int(config.batch_size)
        epochs = int(config.epochs)
        ldist_def = config.latent_distribution
        kappa = 0.0
        alpha = 1.0
        latent_distrib = ldist_def.dist_type
        embedding_source = config.embedding_source
        redundancy_reg_penalty = config.redundancy_reg_penalty
        vocab = self.vocabulary
        if embedding_source.find(':'):
            vocab, _ = self._initialize_vocabulary(embedding_source)
        if latent_distrib == 'vmf':
            kappa = ldist_def.kappa
        elif latent_distrib == 'logistic_gaussian':
            alpha = ldist_def.alpha
        bert_base, _ = nlp.model.get_model('bert_12_768_12',  
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
        model = SeqBowVED(bert_base, vocab,
                          coherence_coefficient=8.0,
                          reporter=None,
                          latent_distribution=latent_distrib,
                          n_latent=n_latent,
                          redundancy_reg_penalty=redundancy_reg_penalty,
                          max_sent_len=self.sent_size,
                          kappa = kappa, 
                          batch_size=batch_size,
                          kld=self.kld_wt, wd_freqs=self.wd_freqs, num_val_words = self.num_val_words,
                          warmup_ratio=self.warmup_ratio,
                          optimizer = optimizer,
                          epochs = epochs,
                          gen_lr = gen_lr,
                          dec_lr = dec_lr,
                          min_lr = min_lr,
                          ctx=ctx,
                          log_interval=self.log_interval)
        return model

    def train_model(self, config, reporter):
        ctx_list = self._get_mxnet_visible_gpus() if self.use_gpu else [mx.cpu()]
        ctx = ctx_list[0]
        seq_ved_model = self._get_ved_model(config, reporter, ctx)
        obj, npmi, perplexity, redundancy = \
            seq_ved_model.fit_with_validation(self.train_data, self.train_labels, self.test_data, self.test_labels)
        return seq_ved_model.model, obj, npmi, perplexity, redundancy


    def write_model(self, m, config, epoch_id=0):
        model_dir = self.model_out_dir
        if model_dir:
            suf = '_'+ str(epoch_id) if epoch_id > 0 else ''
            pfile = os.path.join(model_dir, ('model.params' + suf))
            conf_file = os.path.join(model_dir, ('model.config' + suf))
            vocab_file = os.path.join(model_dir, ('vocab.json' + suf))
            m.save_parameters(pfile)
            dd = {}
            specs = json.dumps(config, sort_keys=True, indent=4)
            with open(conf_file, 'w') as f:
                f.write(specs)
            with open(vocab_file, 'w') as f:
                f.write(m.vocabulary.to_json())

def get_trainer(args):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,
                                                        i_dt.minute,i_dt.second)
    print("Set logging config to {}".format(train_out_dir))
    logging_config(folder=train_out_dir, name='train_trans_vae', level=logging.INFO, no_console=False)
    logging.info(args)
    bow_vocab = load_vocab(args.bow_vocab_file)
    data_train, bert_base, vocab, data_csr = load_dataset_bert(args.tr_file, len(bow_vocab),
                                                               max_len=args.sent_size, ctx=mx.cpu())
    if args.val_file:
        data_val, _, _, val_csr = load_dataset_bert(args.val_file, len(bow_vocab), max_len=args.sent_size, ctx=mx.cpu())
        val_wds = val_csr.sum().asscalar()
    else:
        data_val, val_csr, val_wds = None, None, None
    wd_freqs = get_wd_freqs(data_csr)
    trainer = SeqBowVEDTrainer(
        train_out_dir,
        args.sent_size,
        bow_vocab,
        wd_freqs,
        val_wds, 
        config.warmup_ratio,
        (data_train, data_csr),
        (data_val, val_csr),
        use_gpu = args.use_gpu,
        log_interval = args.log_interval
        )
    return trainer, train_out_dir


def model_select_main(c_args):
    tmnt_config = TMNTConfig(c_args.config_space).get_configspace()
    trainer, log_dir = get_trainer(c_args)
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
    selector.select_model(trainer)
    

def train_main(c_args):
    try:
        with open(c_args.config, 'r') as f:
            config_dict = json.load(f)
    except:
        logging.error("File passed to --config, {}, does not appear to be a valid .json configuration instance".format(c_args.config))
        raise Exception("Invalid JSON configuration file")
    config = ag.space.Dict(**config_dict)    
    trainer, _ = get_trainer(c_args)
    model, obj = trainer.train_with_single_config(config, 1)
    trainer.write_model(model, config_dict)
    
    
            

