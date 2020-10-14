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

from mxnet import gluon

from tmnt.models.base.base_vae import BaseVAE
from tmnt.models.seq_bow.models import BertBowVED
from tmnt.coherence.npmi import EvaluateNPMI

class SeqBowVED(BaseVAE):

    def __init__(self, bert_base, vocab, coherence_coefficient=8.0, reporter=None, latent_distribution="vmf", n_latent=20, redundancy_reg_penalty=0.0, max_sent_len=64, kappa=64.0, batch_size=32, kld=1.0, wd_freqs=None, num_val_words=-1, warmup_ratio=0.1, optimizer="adam", epochs=3, gen_lr=0.000001, dec_lr=0.01, min_lr=0.00000005, ctx=mx.cpu(), log_interval=1, log_method='log'):
        super().__init__(log_method=log_method)
        self.bert_base = bert_base
        self.coherence_coefficient = coherence_coefficient
        self.reporter = reporter
        self.vocabulary = vocab
        self.latent_distribution = latent_distribution
        self.log_interval = log_interval
        self.redundancy_reg_penalty = redundancy_reg_penalty
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
        self.min_lr = min_lr
        self.warmup_ratio = 0.1
        self.weight_decay = 0.00001
        self.offset_factor = 1.0
        self.ctx = ctx
        self.num_val_words = num_val_words
        self.validate_each_epoch = True
        self.max_steps = 2

    def _get_model(self):
        model = BertBowVED(self.bert_base, self.vocabulary, self.latent_distribution, 
                           n_latent=self.n_latent, max_sent_len=self.max_sent_len,
                           kappa = self.kappa, 
                           batch_size = self.batch_size,
                           kld=1.0, wd_freqs=self.wd_freqs,
                           redundancy_reg_penalty=self.redundancy_reg_penalty,
                           ctx=self.ctx)
        return model

    def fit(self, X, y):
        raise NotImplementedError()

    def _compute_coherence(self, model, k, test_data, log_terms=False):
        num_topics = model.n_latent
        sorted_ids = model.get_top_k_terms(k)
        num_topics = min(num_topics, sorted_ids.shape[-1])
        top_k_words_per_topic = [[ int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]
        npmi_eval = EvaluateNPMI(top_k_words_per_topic)
        npmi = npmi_eval.evaluate_csr_mat(test_data)
        unique_term_ids = set()
        unique_limit = 5  ## only consider the top 5 terms for each topic when looking at degree of redundancy
        for i in range(num_topics):
            topic_ids = list(top_k_words_per_topic[i][:unique_limit])
            for j in range(len(topic_ids)):
                unique_term_ids.add(topic_ids[j])
        redundancy = (1.0 - (float(len(unique_term_ids)) / num_topics / unique_limit)) ** 2.0
        logging.info("Test Coherence: {}".format(npmi))
        if log_terms:
            top_k_tokens = [list(map(lambda x: self.vocabulary.idx_to_token[x], list(li))) for li in top_k_words_per_topic]
            for i in range(num_topics):
                logging.info("Topic {}: {}".format(i, top_k_tokens[i]))
        return npmi, redundancy

    
    def _perplexity(self, dataloader, num_words, last_batch_size, num_batches):
        total_rec_loss = 0.0
        total_kl_loss  = 0.0
        for i, seqs in enumerate(dataloader):
            input_ids, valid_length, type_ids, output_vocab = seqs
            _, rec_loss, kl_loss, _, _ = self.model(input_ids.as_in_context(self.ctx), type_ids.as_in_context(self.ctx),
                                                                    valid_length.astype('float32').as_in_context(self.ctx),
                                                                    output_vocab.as_in_context(self.ctx))
            if i == num_batches - 1:
                total_rec_loss += rec_loss[:last_batch_size].sum().asscalar()
                total_kl_loss  += kl_loss[:last_batch_size].sum().asscalar()
            else:
                total_rec_loss += rec_loss.sum().asscalar()
                total_kl_loss  += kl_loss.sum().asscalar()
        ll = (total_rec_loss + total_kl_loss) / num_words
        if ll < 709.0:
            perplexity = math.exp(ll)
        else:
            perplexity = 1e300
        return perplexity


    def validate(self, model, bow_val_X, dataloader):
        last_batch_size = bow_val_X.shape[0] % self.batch_size
        if last_batch_size > 0:
            num_batches = (bow_val_X.shape[0] // self.batch_size) + 1
        else:
            num_batches = bow_val_X.shape[0] // self.batch_size
            last_batch_size = self.batch_size
        ppl = self._perplexity(dataloader, self.num_val_words, last_batch_size, num_batches)
        return ppl
    

    def fit_with_validation(self, X, y, val_X, val_y):
        seq_train, bow_train = X
        model = self._get_model()
        self.model = model
        dataloader = mx.gluon.data.DataLoader(seq_train, batch_size=self.batch_size,
                                              shuffle=True, last_batch='rollover')
        if val_X is not None:
            seq_val, bow_val = val_X
            dataloader_val = mx.gluon.data.DataLoader(seq_val, batch_size=self.batch_size, last_batch='rollover',
                                                       shuffle=False)

        num_train_examples = len(seq_train)
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

        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0                
        for epoch_id in range(self.epochs):
            step_loss = 0
            step_recon_ls = 0
            step_kl_ls = 0
            step_red_ls = 0
            for batch_id, seqs in enumerate(dataloader):
                step_num += 1
                if step_num < num_warmup_steps:
                    new_lr = lr * step_num / num_warmup_steps
                else:
                    offset = (step_num - num_warmup_steps) * lr / ((num_train_steps - num_warmup_steps) * self.offset_factor)
                    new_lr = max(lr - offset, self.min_lr)
                gen_trainer.set_learning_rate(new_lr)
                with mx.autograd.record():
                    input_ids, valid_length, type_ids, output_vocab = seqs
                    ls, recon_ls, kl_ls, redundancy_ls, predictions = model(input_ids.as_in_context(self.ctx), type_ids.as_in_context(self.ctx),
                                                                            valid_length.astype('float32').as_in_context(self.ctx),
                                                                            output_vocab.as_in_context(self.ctx))
                    ls = ls.mean()
                ls.backward()
                grads = [p.grad(self.ctx) for p in differentiable_params]
                gluon.utils.clip_global_norm(grads, 1)
                lat_trainer.step(1)
                dec_trainer.step(1) # update decoder trainer associated weights
                gen_trainer.step(1) # step of 1 since we averaged loss over batch
                step_loss += ls.asscalar()
                step_recon_ls += recon_ls.mean().asscalar()
                step_kl_ls += kl_ls.mean().asscalar()
                step_red_ls += redundancy_ls.mean().asscalar()
                if (batch_id + 1) % (self.log_interval) == 0:
                    logging.info('[Epoch {}/{} Batch {}/{}] loss={:.4f}, recon_loss={:.4f}, kl_loss={:.4f}, red_loss={:.4f}, gen_lr={:.7f}'
                                 .format(epoch_id, self.epochs, batch_id + 1, len(dataloader),
                                         step_loss / self.log_interval, step_recon_ls / self.log_interval,
                                         step_kl_ls / self.log_interval, step_red_ls / self.log_interval,
                                         gen_trainer.learning_rate))
                    step_loss = 0
                    step_recon_ls = 0
                    step_kl_ls = 0
                    _, _ = self._compute_coherence(model, 10, bow_train, log_terms=True)
            if val_X is not None and (self.validate_each_epoch or epoch_id == self.epochs-1):
                npmi, redundancy = self._compute_coherence(model, 10, bow_train, log_terms=True)
                ppl = self.validate(model, bow_val, dataloader_val)
                obj = (npmi - redundancy) * self.coherence_coefficient - ( ppl / 1000 )
                b_obj = max(min(obj, 100.0), -100.0)
                sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
                self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                                    .format(epoch_id, sc_obj, ppl, npmi, redundancy))
                if self.reporter:
                    self.reporter(epoch=epoch_id+1, objective=sc_obj, time_step=time.time(), coherence=npmi,
                                  perplexity=ppl, redundancy=redundancy)
        return sc_obj, npmi, ppl, redundancy
