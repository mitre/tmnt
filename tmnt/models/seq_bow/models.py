# codeing: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

__all__ = ['TransformerBowVED']

import math
import os
import numpy as np

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock, Block

from tmnt.distributions import LogisticGaussianLatentDistribution, GaussianLatentDistribution
from tmnt.distributions import GaussianUnitVarLatentDistribution, HyperSphericalLatentDistribution
from tmnt.models.seq_seq.trans_seq_models import TransformerEncoder

class TransformerBowVED(Block):

    def __init__(self, bow_vocab_size, vocabulary, emb_dim, latent_distrib='vmf', num_units=512, hidden_size=512, num_heads=4,
                 n_latent=256, max_sent_len=32, transformer_layers=2, 
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_freqs=None,
                 ctx = mx.cpu(),
                 prefix=None, params=None):
        super(TransformerBowVED, self).__init__(prefix=prefix, params=params)
        self.kld_wt = kld
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.max_sent_len = max_sent_len
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.wd_embed_dim = emb_dim
        self.vocab_size = len(vocabulary.idx_to_token)
        self.bow_vocab_size = bow_vocab_size
        self.latent_distrib = latent_distrib
        self.num_units = num_units
        self.hidden_size = hidden_size        
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.kappa = kappa
        with self.name_scope():
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=kappa, ctx=self.model_ctx, dr=0.0)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarLatentDistribution(n_latent, ctx, dr=0.0, var=0.05)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
            self.embedding = nn.Embedding(self.vocab_size, self.wd_embed_dim)
            self.encoder = TransformerEncoder(self.wd_embed_dim, self.num_units, hidden_size=hidden_size, num_heads=num_heads,
                                              n_layers=transformer_layers, n_latent=n_latent, sent_size = max_sent_len,
                                              batch_size = batch_size, ctx = ctx)
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.bow_vocab_size, activation=None)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        if self.vocabulary.embedding is not None:
            emb = vocabulary.embedding.idx_to_vec
            emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=1) + 1e-10
            emb_norm = emb / emb_norm_val
            self.embedding.weight.set_data(emb_norm)
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1
            total = freq_nd.sum()
            log_freq = freq_nd.log() - freq_nd.sum().log()
            bias_param = self.decoder.collect_params().get('bias')
            bias_param.set_data(log_freq)
            bias_param.grad_req = 'null'
            self.out_bias = bias_param.data()

    def get_top_k_terms(self, k):
        """
        Returns the top K terms for each topic based on sensitivity analysis. Terms whose 
        probability increases the most for a unit increase in a given topic score/probability
        are those most associated with the topic. This is just the topic-term weights for a 
        linear decoder - but code here will work with arbitrary decoder.
        """
        z = mx.nd.ones(shape=(1, self.n_latent), ctx=self.model_ctx)
        jacobian = mx.nd.zeros(shape=(self.bow_vocab_size, self.n_latent), ctx=self.model_ctx)
        z.attach_grad()        
        for i in range(self.bow_vocab_size):
            with mx.autograd.record():
                y = self.decoder(z)
                yi = y[0][i]
            yi.backward()
            jacobian[i] = z.grad
        sorted_j = jacobian.argsort(axis=0, is_ascend=False)
        return sorted_j

        

    def __call__(self, wp_toks, bow):
        return super(TransformerBowVED, self).__call__(wp_toks, bow)

    def set_kl_weight(self, epoch, max_epochs):
        burn_in = int(max_epochs / 10)
        eps = 1e-6
        if epoch > burn_in:
            self.kld_wt = ((epoch - burn_in) / (max_epochs - burn_in)) + eps
        else:
            self.kld_wt = eps
        return self.kld_wt

    def encode(self, toks):
        embedded = self.embedding(toks)
        enc = self.encoder(embedded)
        return self.latent_dist.mu_encoder(enc)

    def forward(self, toks, bow):
        embedded = self.embedding(toks)
        enc = self.encoder(embedded)
        z, KL = self.latent_dist(enc, self.batch_size)
        y = self.decoder(z)
        y = mx.nd.softmax(y, axis=1)
        rr = bow * mx.nd.log(y+1e-12)
        recon_loss = -mx.nd.sparse.sum( rr, axis=1 )
        KL_loss = ( KL * self.kld_wt )
        loss = recon_loss + KL_loss
        return loss, recon_loss, KL_loss, y


class BertBowVED(Block):
    def __init__(self, bert_base, bow_vocab_size, latent_distrib='vmf', 
                 n_latent=256, max_sent_len=32, 
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_freqs=None,
                 ctx = mx.cpu(),
                 prefix=None, params=None):
        super(BertBowVED, self).__init__(prefix=prefix, params=params)
        self.kld_wt = kld
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.max_sent_len = max_sent_len
        self.batch_size = batch_size
        self.bow_vocab_size = bow_vocab_size
        self.latent_distrib = latent_distrib
        self.kappa = kappa
        with self.name_scope():
            self.encoder = bert_base            
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=kappa, ctx=self.model_ctx, dr=0.0)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarLatentDistribution(n_latent, ctx, dr=0.0, var=0.05)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.bow_vocab_size, activation=None)
        self.latent_dist.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        self.decoder.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1
            total = freq_nd.sum()
            log_freq = freq_nd.log() - freq_nd.sum().log()
            bias_param = self.decoder.collect_params().get('bias')
            bias_param.set_data(log_freq)
            bias_param.grad_req = 'null'
            self.out_bias = bias_param.data()

    def get_top_k_terms(self, k):
        """
        Returns the top K terms for each topic based on sensitivity analysis. Terms whose 
        probability increases the most for a unit increase in a given topic score/probability
        are those most associated with the topic. This is just the topic-term weights for a 
        linear decoder - but code here will work with arbitrary decoder.
        """
        z = mx.nd.ones(shape=(1, self.n_latent), ctx=self.model_ctx)
        jacobian = mx.nd.zeros(shape=(self.bow_vocab_size, self.n_latent), ctx=self.model_ctx)
        z.attach_grad()        
        for i in range(self.bow_vocab_size):
            with mx.autograd.record():
                y = self.decoder(z)
                yi = y[0][i]
            yi.backward()
            jacobian[i] = z.grad
        sorted_j = jacobian.argsort(axis=0, is_ascend=False)
        return sorted_j

    def __call__(self, toks, tok_types, valid_length, bow):
        return super(BertBowVED, self).__call__(toks, tok_types, valid_length, bow)

    def set_kl_weight(self, epoch, max_epochs):
        burn_in = int(max_epochs / 10)
        eps = 1e-6
        if epoch > burn_in:
            self.kld_wt = ((epoch - burn_in) / (max_epochs - burn_in)) + eps
        else:
            self.kld_wt = eps
        return self.kld_wt

    def forward(self, toks, tok_types, valid_length, bow):
        _, enc = self.encoder(toks, tok_types, valid_length)
        z, KL = self.latent_dist(enc, self.batch_size)
        y = self.decoder(z)
        y = mx.nd.softmax(y, axis=1)
        rr = bow * mx.nd.log(y+1e-12)
        recon_loss = -mx.nd.sparse.sum( rr, axis=1 )
        KL_loss = ( KL * self.kld_wt )
        loss = recon_loss + KL_loss
        return loss, recon_loss, KL_loss, y
    

class TransformerBowVEDTest(Block):

    def __init__(self, bow_vocab_size, vocabulary, emb_dim, latent_distrib='vmf', num_units=512, hidden_size=512, num_heads=4,
                 n_latent=256, max_sent_len=32, transformer_layers=2, 
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_freqs=None,
                 ctx = mx.cpu(),
                 prefix=None, params=None):
        super(TransformerBowVEDTest, self).__init__(prefix=prefix, params=params)
        self.kld_wt = kld
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.max_sent_len = max_sent_len
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.wd_embed_dim = emb_dim
        self.vocab_size = len(vocabulary.idx_to_token)
        self.bow_vocab_size = bow_vocab_size
        self.latent_distrib = latent_distrib
        self.num_units = num_units
        self.hidden_size = hidden_size        
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.kappa = kappa
        with self.name_scope():
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=kappa, ctx=self.model_ctx, dr=0.0)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarLatentDistribution(n_latent, ctx, dr=0.0, var=0.05)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
            self.embedding = nn.Dense(in_units=self.bow_vocab_size, units=self.wd_embed_dim, activation='tanh')
            self.encoder = nn.Dense(in_units=self.wd_embed_dim, units=200, activation='softrelu')
            #self.encoder = TransformerEncoder(self.wd_embed_dim, self.num_units, hidden_size=hidden_size, num_heads=num_heads,
            #                                  n_layers=transformer_layers, n_latent=n_latent, sent_size = max_sent_len,
            #                                  batch_size = batch_size, ctx = ctx)
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.bow_vocab_size, activation=None)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        self.latent_dist.post_init(self.model_ctx)
        if self.vocabulary.embedding is not None:
            emb = vocabulary.embedding.idx_to_vec
            emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=1) + 1e-10
            emb_norm = emb / emb_norm_val
            self.embedding.weight.set_data(emb_norm)
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1
            total = freq_nd.sum()
            log_freq = freq_nd.log() - freq_nd.sum().log()
            bias_param = self.decoder.collect_params().get('bias')
            bias_param.set_data(log_freq)
            bias_param.grad_req = 'null'
            self.out_bias = bias_param.data()


    def __call__(self, wp_toks, bow):
        return super(TransformerBowVEDTest, self).__call__(wp_toks, bow)

    def forward(self, toks, bow):
        embedded = self.embedding(bow)
        enc = self.encoder(embedded)
        z, KL = self.latent_dist(enc, self.batch_size)
        y = self.decoder(z)
        y = mx.nd.softmax(y, axis=1)
        rr = bow * mx.nd.log(y+1e-12)
        recon_loss = -mx.nd.sparse.sum( rr, axis=1 )
        loss = recon_loss + KL
        return loss, recon_loss, KL, y
    
