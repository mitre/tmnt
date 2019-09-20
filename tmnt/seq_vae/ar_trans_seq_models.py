# codeing: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""

__all__ = ['ARTransformerVAE']


import math
import os
import numpy as np

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock, Block
from gluonnlp.model import TransformerEncoderCell, TransformerEncoder
from tmnt.distributions import LogisticGaussianLatentDistribution, GaussianLatentDistribution, HyperSphericalLatentDistribution
from gluonnlp.model.transformer import TransformerDecoder

class ARTransformerVAE(Block):

        
        def __init__(self, bert_base, latent_distrib='vmf',
                     wd_embed_dim=300, num_units=512, hidden_size=512, n_latent=256, max_sent_len=64, transformer_layers=6,
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_temp=0.01, ctx = mx.cpu(),
                 increasing=True, decreasing=False,
                 prefix=None, params=None):
            super(BertTransVAE, self).__init__(prefix=prefix, params=params)
            self.kld_wt = kld
            self.bert = bert_base
            self.n_latent = n_latent
            self.model_ctx = ctx
            self.max_sent_len = max_sent_len
            self.batch_size = batch_size
            self.wd_embed_dim = wd_embed_dim
            self.latent_distrib = latent_distrib
            with self.name_scope():
                if latent_distrib == 'logistic_gaussian':
                        self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx, dr=0.0)
                elif latent_distrib == 'vmf':
                        self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=kappa, dr=0.0, ctx=self.model_ctx)
                elif latent_distrib == 'gaussian':
                        self.latent_dist = GaussianLatentDistribution(n_latent, ctx, dr=0.0)
                elif latent_distrib == 'gaussian_unitvar':
                        self.latent_dist = GaussianUnitVarLatentDistribution(n_latent, ctx, dr=0.0)
                else:
                        raise Exception("Invalid distribution ==> {}".format(latent_distrib))
                self.decoder = TransformerDecoder(units=num_units, hidden_size=hidden_size,
                                              num_layers=transformer_layers, n_latent=n_latent, max_length = max_sent_len,
                                                  tx = ctx)
                self.vocab_size = self.bert.word_embed[0].params.get('weight').shape[0]
                self.out_embedding = gluon.nn.Embedding(input_dim=self.vocab_size, output_dim=wd_embed_dim, weight_initializer=mx.init.Uniform(0.1))
                self.inv_embed = InverseEmbed(batch_size, max_sent_len, self.wd_embed_dim, temp=wd_temp, ctx=self.model_ctx, params = self.out_embedding.params)
                self.ce_loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=-1, from_logits=True)


        def decode_seq(self, inputs, states, valid_length=None):
                outputs, states, additional_outputs = self.decoder.decode_seq(inputs=self.tgt_embed(inputs),
                                                                              states=states,
                                                                              valid_length=valid_length)
                outputs = self.tgt_proj(outputs)
                return outputs, states, additional_outputs


        def forward(self, toks):
                embedded = self.embedding(toks)
                enc = self.encoder(embedded)
                z, KL = self.latent_dist(enc, self.batch_size)
                decoder_states = self.decoder.init_state_from_encoder(z)
                outputs, _, _ = self.decode
                #y = self.decoder(z)
                #prob_logits = self.inv_embed(y)
                #log_prob = mx.nd.log_softmax(prob_logits)
                #recon_loss = self.ce_loss_fn(log_prob, toks)
                #kl_loss = (KL * self.kld_wt)
                #loss = recon_loss + kl_loss
                return loss, recon_loss, kl_loss, log_prob


