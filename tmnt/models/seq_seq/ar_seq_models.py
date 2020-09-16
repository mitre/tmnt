# codeing: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""

__all__ = ['ARTransformerVAE']


import math
import os
import numpy as np
import datetime
import logging

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock, Block
from tmnt.models.seq_seq.seq_data_loader import load_dataset_bert, load_dataset_basic
from gluonnlp.model import TransformerEncoderCell
from tmnt.utils.log_utils import logging_config
from tmnt.distributions import LogisticGaussianLatentDistribution, GaussianLatentDistribution, HyperSphericalLatentDistribution
from tmnt.models.seq_seq.trans_seq_models import TransformerEncoder, InverseEmbed
from gluonnlp.model.transformer import TransformerDecoder


class ARTransformerVAE(Block):
    def __init__(self, vocabulary, emb_dim, latent_distrib='vmf', num_units=512, hidden_size=512, num_heads=4,
                 n_latent=256, max_sent_len=64, transformer_layers=6, label_smoothing_epsilon=0.0,
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_temp=0.01,
                 ctx = mx.cpu(),
                 prefix=None, params=None):

        super(ARTransformerVAE, self).__init__(prefix=prefix, params=params)
        self.kld_wt = kld
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.max_sent_len = max_sent_len
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.wd_embed_dim = emb_dim
        self.vocab_size = len(vocabulary.idx_to_token)
        self.latent_distrib = latent_distrib
        self.num_units = num_units
        self.hidden_size = hidden_size        
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.label_smoothing_epsilon = label_smoothing_epsilon
        self.kappa = kappa

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
            self.embedding = nn.Embedding(self.vocab_size, self.wd_embed_dim)
            self.encoder = TransformerEncoder(self.wd_embed_dim, self.num_units, hidden_size=hidden_size, num_heads=num_heads,
                                              n_layers=transformer_layers, n_latent=n_latent, sent_size = max_sent_len,
                                              batch_size = batch_size, ctx = ctx)

            
            #self.decoder = TransformerDecoder(units=num_units, hidden_size=hidden_size,
            #                                  num_layers=transformer_layers, n_latent=n_latent, max_length = max_sent_len,
            #                                  tx = ctx)
            self.decoder = TransformerDecoder(num_layers=transformer_layers,
                                 num_heads=num_heads,
                                 max_length=max_sent_len,
                                 units=self.num_units,
                                 hidden_size=hidden_size,
                                 dropout=0.1,
                                 scaled=True,
                                 use_residual=True,
                                 weight_initializer=None,
                                 bias_initializer=None,
                                 prefix='transformer_' + 'dec_', params=params)
            self.inv_embed = InverseEmbed(batch_size, max_sent_len, self.wd_embed_dim, temp=wd_temp, ctx=self.model_ctx,
                                          params = self.embedding.params)
            self.ce_loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=-1, from_logits=True)
        self.embedding.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
        if self.vocabulary.embedding:
            self.embedding.weight.set_data(self.vocabulary.embedding.idx_to_vec)


    def decode_seq(self, inputs, states, valid_length=None):
        outputs, states, additional_outputs = self.decoder.decode_seq(inputs=self.embedding(inputs),
                                                                      states=states,
                                                                      valid_length=valid_length)
        outputs = self.inv_embed(outputs)
        return outputs, states, additional_outputs


    def decode_step(self, step_input, states):
        step_output, states, step_additional_outputs =\
            self.decoder(self.embedding(step_input), states)
        step_output = self.inv_embed(step_output)
        return step_output, states, step_additional_outputs


    def encode(self, inputs):
        embedded = self.embedding(toks)
        enc = self.encoder(embedded)
        z, KL = self.latent_dist(enc, self.batch_size)
        return z, KL

    def forward(self, toks):
        z, KL = self.encode(toks)
        decoder_states = self.decoder.init_state_from_encoder(z)
        outputs, _, _ = self.decoder_seq(toks, decoder_states)
        #y = self.decoder(z)
        #prob_logits = self.inv_embed(y)
        #log_prob = mx.nd.log_softmax(prob_logits)
        #recon_loss = self.ce_loss_fn(log_prob, toks)
        #kl_loss = (KL * self.kld_wt)
        #loss = recon_loss + kl_loss
        #return loss, recon_loss, kl_loss, log_prob
        return outputs

def test_ar(args):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
    print("Set logging config to {}".format(train_out_dir))
    logging_config(folder=train_out_dir, name='train_trans_vae', level=logging.INFO, no_console=False)
    logging.info(args)
    context = mx.cpu() if args.gpus is None or args.gpus == '' else mx.gpu(int(args.gpus))
    emb = nlp.embedding.create('glove', source = args.embedding_source) if args.embedding_source else None
    data_train, vocab = load_dataset_basic(args.input_file, vocab=None, json_text_key=args.json_text_key, max_len=args.sent_size,
                                           max_vocab_size=args.max_vocab_size, ctx=context)
    if emb:
        vocab.set_embedding(emb)
        _, emb_size = vocab.embedding.idx_to_vec.shape
        oov_items = 0
        for word in vocab.embedding._idx_to_token:
            if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                oov_items += 1
                vocab.embedding[word] = mx.nd.random.normal(0.0, 0.1, emb_size)
        logging.info("** There are {} out of vocab items **".format(oov_items))
    else:
        logging.info("** No pre-trained embedding provided, learning embedding weights from scratch **")
    emb_dim = len(vocab.embedding.idx_to_vec[0])
    model = ARTransformerVAE(vocab, emb_dim, args.latent_dist, num_units=args.num_units, hidden_size=args.hidden_size,
                               num_heads=args.num_heads,
                               n_latent=args.latent_dim, max_sent_len=args.sent_size,
                               transformer_layers=args.transformer_layers,
                               kappa = args.kappa, 
                               batch_size=args.batch_size,
                               kld=args.kld_wt, ctx=context)
    model.latent_dist.initialize(init=mx.init.Xavier(magnitude=2.34), ctx=context)
    model.encoder.initialize(init=mx.init.Xavier(magnitude=2.34), ctx=context)
    #model.decoder.initialize(init=mx.init.Xavier(magnitude=2.34), ctx=context)
    pad_id = vocab[vocab.padding_token]
    




