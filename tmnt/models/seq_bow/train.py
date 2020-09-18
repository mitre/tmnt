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
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from tmnt.coherence.npmi import EvaluateNPMI
from tmnt.models.bow.bow_doc_loader import load_vocab
from tmnt.models.seq_bow.models import TransformerBowVED, TransformerBowVEDTest, BertBowVED
from tmnt.utils.log_utils import logging_config
from tmnt.models.seq_bow.sb_data_loader import load_dataset_basic_seq_bow, load_dataset_bert

__all__ = ['train_main']

def get_wd_freqs(data_csr, max_sample_size=1000000):
    sample_size = min(max_sample_size, data_csr.shape[0])
    data = data_csr[:sample_size] 
    sums = mx.nd.sum(data, axis=0)
    return sums


def get_basic_model(args, bow_vocab_size, vocab, emb_dim, wd_freqs, ctx):
    model = TransformerBowVED(bow_vocab_size, vocab, emb_dim, args.latent_dist, num_units=args.num_units,
                              hidden_size=args.hidden_size,
                              num_heads=args.num_heads,
                              n_latent=args.latent_dim, max_sent_len=args.sent_size,
                              transformer_layers=args.transformer_layers,
                              kappa = args.kappa, 
                              batch_size=args.batch_size,
                              wd_freqs=wd_freqs,
                              kld=args.kld_wt, ctx=ctx)
    return model


def get_bert_model(args, bert_base, bow_vocab_size, wd_freqs, ctx):
    model = BertBowVED(bert_base, bow_vocab_size, args.latent_dist, 
                              n_latent=args.latent_dim, max_sent_len=args.sent_size,
                              kappa = args.kappa, 
                              batch_size=args.batch_size,
                              kld=args.kld_wt, wd_freqs=wd_freqs, ctx=ctx)
    return model


def compute_coherence(model, bow_vocab, k, test_data, log_terms=False, ctx=mx.cpu()):
    num_topics = model.n_latent
    sorted_ids = model.get_top_k_terms(k)
    num_topics = min(num_topics, sorted_ids.shape[-1])
    top_k_words_per_topic = [[ int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]
    npmi_eval = EvaluateNPMI(top_k_words_per_topic)
    npmi = npmi_eval.evaluate_csr_mat(test_data)
    logging.info("Test Coherence: {}".format(npmi))
    if log_terms:
        top_k_tokens = [list(map(lambda x: bow_vocab.idx_to_token[x], list(li))) for li in top_k_words_per_topic]
        for i in range(num_topics):
            logging.info("Topic {}: {}".format(i, top_k_tokens[i]))
    return npmi


class SeqBowVEDTrainer(BaseTrainer):
    def __init__(self, model_out_dir, sent_size, vocabulary, wd_freqs, train_data, 
                 test_data, train_labels=None, test_labels=None, use_gpu=False, val_each_epoch=True, rng_seed=1234):
        super().__init__(train_data, test_data, train_labels, test_labels, rng_seed)
        self.model_out_dir = model_out_dir
        self.use_gpu = use_gpu
        self.vocabulary = vocabulary
        self.wd_freqs = wd_freqs
        self.vocab_cache = {}
        self.validate_each_epoch = val_each_epoch
        self.seed_matrix = None
        self.sent_size = sent_size
        self.kld_wt = 1.0

    def _get_ved_model(self, config, reporter, ctx):
        gen_lr = config.gen_lr
        dec_lr = config.dec_lr
        latent_distrib = config.latent_distribution
        optimizer = config.optimizer
        n_latent = int(config.n_latent)
        batch_size = int(config.batch_size)
        epochs = int(config.epochs)
        ldist_def = config.latent_distribution
        kappa = 0.0
        alpha = 1.0
        latent_distrib = ldist_def.dist_type
        if latent_distrib == 'vmf':
            kappa = ldist_def.kappa
        elif latent_distrib == 'logistic_gaussian':
            alpha = ldist_def.alpha
        model = SeqBowVED(self.bert_base, len(self.vocabulary), latent_distrib, 
                          n_latent=n_latent, max_sent_len=self.sent_size,
                          kappa = kappa, 
                          batch_size=batch_size,
                          kld=self.kld_wt, wd_freqs=self.wd_freqs,
                          warmup_ratio=self.warmup_ratio,
                          optimizer = optimizer,
                          epochs = epochs,
                          gen_lr = gen_lr,
                          dec_lr = dec_lr,
                          ctx=ctx)
        return model

    def train_model(self, config, reporter):
        ctx_list = get_mxnet_visible_gpus() if self.use_gpu else [mx.cpu()]
        ctx = ctx_list[0]
        seq_ved_model = self._get_ved_model(config, reporter, ctx)
        obj, npmi, perplexity, redundancy = seq_ved_model.fit_with_validation(self.train_data, self.train_labels, self.test_data, self.test_labels)
        return seq_ved_model.model, obj, npmi, perplexity, redundancy


    def write_model(m, args, epoch_id=0):
        model_dir = self.model_out_dir
        if model_dir:
            suf = '_'+ str(epoch_id) if epoch_id > 0 else ''
            pfile = os.path.join(model_dir, ('model.params' + suf))
            conf_file = os.path.join(model_dir, ('model.config' + suf))
            vocab_file = os.path.join(model_dir, ('vocab.json' + suf))
            m.save_parameters(pfile)
            dd = {}
            dd['latent_dist'] = m.latent_distrib
            dd['num_units'] = m.num_units
            dd['num_heads'] = m.num_heads        
            dd['hidden_size'] = m.hidden_size
            dd['n_latent'] = m.n_latent
            dd['transformer_layers'] = m.transformer_layers
            dd['kappa'] = m.kappa
            dd['sent_size'] = m.max_sent_len
            dd['embedding_size'] = m.wd_embed_dim
            specs = json.dumps(dd)
            with open(conf_file, 'w') as f:
                f.write(specs)
            with open(vocab_file, 'w') as f:
                f.write(m.vocabulary.to_json())


def train_main(args):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,
                                                        i_dt.minute,i_dt.second)
    print("Set logging config to {}".format(train_out_dir))
    logging_config(folder=train_out_dir, name='train_trans_vae', level=logging.INFO, no_console=False)
    logging.info(args)
    context = mx.cpu() if args.gpus is None or args.gpus == '' else mx.gpu(int(args.gpus))
    bow_vocab = load_vocab(args.bow_vocab_file)
    data_train, bert_base, vocab, data_csr = load_dataset_bert(args.input_file, len(bow_vocab),
                                                               max_len=args.sent_size, ctx=context)
    wd_freqs = get_wd_freqs(data_csr)
    model = get_bert_model(args, bert_base, len(bow_vocab), wd_freqs, context)
    pad_id = vocab[vocab.padding_token]
    trainer = SeqBowVEDTrainer(
        train_out_dir,
        args.sent_size,
        vocab,
        wd_freqs,
        data_train,
        data_train
        )
    try:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    except:
        logging.error("File passed to --config, {}, does not appear to be a valid .json configuration instance".format(args.config))
        raise Exception("Invalid JSON configuration file")
    config = ag.space.Dict(**config_dict)
    model, obj = trainer.train_with_single_config(config, 1)
    trainer.write_model(model, config_dict)
    
    
            

