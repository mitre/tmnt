# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import math
import logging
import time
import io
import os
import psutil
import mxnet as mx
import numpy as np
import random

from tmnt.models.base.base_vae import BaseVAE

class SeqBowVED(BaseVAE):

    def __init__(self, bert_base, vocab_len, latent_distribution, n_latent, max_sent_len, kappa, batch_size, kld, wd_freqs, warmup_ratio, optimizer, epochs, gen_lr, dec_lr, ctx, log_method='log'):
        super().__init__(log_method=log_method)
        self.bert_base = bert_base
        self.vocab_len = vocab_len
        self.latent_distribution = latent_distribution
        self.log_interval = 1
        self.n_latent = n_latent
        self.max_sent_len = max_sent_len
        self.kappa = kappa
        self.batch_size = batch_size
        self.kld = kld
        self.wd_freqs = wd_freqs
        self.optimizer = optimizer
        self.epochs = epochs
        self.gen_lr = gen_lr
        self.dec_lr = dec_lr
        self.warmup_ratio
        self.weight_decay = 0.00001
        self.offset_factor = 1.0
        


    def _get_model(self):
        raise NotImplementedError()

    def fit(self, X, y):
        raise NotImplementedError()

    def fit_with_validation(self, X, y, val_X, val_y):
        #def train_bow_seq_ved(args, model, bow_vocab, data_train, train_csr, data_test=None, ctx=mx.cpu(), use_bert=False):
        dataloader = mx.gluon.data.DataLoader(X, batch_size=self.batch_size,
                                              shuffle=True, last_batch='rollover')
        if data_test:
            dataloader_test = mx.gluon.data.DataLoader(val_X, batch_size=self.batch_size,
                                                       shuffle=False) if data_test else None

        num_train_examples = len(X)
        num_train_steps = int(num_train_examples / self.batch_size * self.epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        step_num = 0
        differentiable_params = []

        lr = self.gen_lr

        gen_trainer = gluon.Trainer(model.encoder.collect_params(), self.optimizer,
                                {'learning_rate': self.gen_lr, 'epsilon': 1e-6, 'wd':self.weight_decay})
        lat_trainer = gluon.Trainer(model.latent_dist.collect_params(), 'adam', {'learning_rate': self.dec_lr, 'epsilon': 1e-6})
        dec_trainer = gluon.Trainer(model.decoder.collect_params(), 'adam', {'learning_rate': self.dec_lr, 'epsilon': 1e-6})    

        # Do not apply weight decay on LayerNorm and bias terms
        for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

        for p in model.encoder.collect_params().values():
            if p.grad_req != 'null':
                differentiable_params.append(p)

        for epoch_id in range(self.epochs):
            step_loss = 0
            step_recon_ls = 0
            step_kl_ls = 0
            for batch_id, seqs in enumerate(dataloader):
                step_num += 1
                if step_num < num_warmup_steps:
                    new_lr = lr * step_num / num_warmup_steps
                else:
                    offset = (step_num - num_warmup_steps) * lr / ((num_train_steps - num_warmup_steps) * self.offset_factor)
                    new_lr = max(lr - offset, self.min_lr)
                gen_trainer.set_learning_rate(new_lr)
                with mx.autograd.record():
                    if use_bert:
                        input_ids, valid_length, type_ids, output_vocab = seqs
                        ls, recon_ls, kl_ls, predictions = model(input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                                                                 valid_length.astype('float32').as_in_context(ctx),
                                                                 output_vocab.as_in_context(ctx))
                    else:
                        input_ids, output_vocab = seqs
                        ls, recon_ls, kl_ls, predictions = model(input_ids.as_in_context(ctx), output_vocab.as_in_context(ctx))
                    ls = ls.mean()
                ls.backward()
                grads = [p.grad(ctx) for p in differentiable_params]
                gluon.utils.clip_global_norm(grads, 1)
                lat_trainer.step(1)
                dec_trainer.step(1) # update decoder trainer associated weights
                gen_trainer.step(1) # step of 1 since we averaged loss over batch
                step_loss += ls.asscalar()
                step_recon_ls += recon_ls.mean().asscalar()
                step_kl_ls += kl_ls.mean().asscalar()
                if (batch_id + 1) % (self.log_interval) == 0:
                    logging.info('[Epoch {}/{} Batch {}/{}] loss={:.4f}, recon_loss={:.4f}, kl_loss={:.4f}, gen_lr={:.7f}'
                                 .format(epoch_id, args.epochs, batch_id + 1, len(dataloader),
                                         step_loss / args.log_interval, step_recon_ls / args.log_interval,
                                         step_kl_ls / args.log_interval,
                                         gen_trainer.learning_rate))
                    step_loss = 0
                    step_recon_ls = 0
                    step_kl_ls = 0
                    _ = compute_coherence(model, bow_vocab, 10, train_csr, log_terms=True)
