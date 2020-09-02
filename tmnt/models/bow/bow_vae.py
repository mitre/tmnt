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
import copy
import random

from mxnet import autograd
from mxnet import gluon
import gluonnlp as nlp
from pathlib import Path

from tmnt.models.bow.bow_doc_loader import DataIterLoader
from tmnt.models.bow.bow_models import BowNTM, MetaDataBowNTM
from tmnt.models.base.base_vae import BaseVAE

from tmnt.coherence.npmi import EvaluateNPMI


MAX_DESIGN_MATRIX = 250000000 

class BaseBowVAE(BaseVAE):
    """
    Bag of words variational autoencoder algorithm

    Parameters
    ----------
    vocabulary: list[string]
        Vocabulary list used for pretrained emeddings

    lr: float, optional (default=0.005)
        Learning rate of training.

    latent_distribution: 'logistic_gaussian' | 'vmf' | 'gaussian' | 'gaussian_unitvar', optional (default="vmf")
        Latent distribution of the variational autoencoder.
    
    n_latent: int, optional (default=20)
        Size of the latent distribution.

    kappa: float, optional (default=64.0)
        Distribution parameter for Von-Mises Fisher distribution, ignored if latent_distribution not 'vmf'.

    alpha: float, optional (default=1.0)
        Prior parameter for Logistic Gaussian distribution, ignored if latent_distribution not 'logistic_gaussian'.

    enc_hidden_dim: int, optional (default=150)
        Size of hidden encoder layers.
    
    coherence_reg_penalty: float, optional (default=0.0)
        Regularization penalty for topic coherence.

    redundancy_reg_penalty: float, optional (default=0.0)
        Regularization penalty for topic redundancy.

    batch_size: int, optional (default=128)
        Batch training size.

    embedding_source: 'random' | 'glove' | 'fasttext' | 'word2vec', optional (default='random')
        Word embedding source for vocabulary.

    embedding_size: int, optional (default=128)
        Word embedding size, ignored if embedding_source not 'random'.

    fixed_embedding: bool, optional(default=False)
        Enable fixed embeddings.

    num_enc_layers: int, optional(default=1)
        Number of layers in encoder.

    enc_dr: float, optional(default=0.1)
        Dropout probability in encoder.

    seed_matrix: mxnet matrix, optional(default=None)
        Seed matrix for guided topic model.

    hybridize: bool, optional(default=False)
        Hybridize underlying mxnet model.

    epochs: int, optional(default=40)
        Number of training epochs.

    gpu_id: int, optional(default=-1)
        ID of GPU device, GPU training disabled if gpu_id<0.
    """
    def __init__(self, vocabulary, coherence_coefficient=8.0, reporter=None, num_val_words=-1, wd_freqs=None, ctx=mx.cpu(), lr=0.005, latent_distribution="vmf", optimizer="adam", n_latent=20, kappa=64.0, alpha=1.0, enc_hidden_dim=150, coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0, batch_size=128, embedding_source="random", embedding_size=128, fixed_embedding=False, num_enc_layers=1, enc_dr=0.1, seed_matrix=None, hybridize=False, epochs=40):
        self.reporter = reporter
        self.coherence_coefficient = coherence_coefficient
        self.lr = lr
        self.latent_distrib = latent_distribution
        self.optimizer = optimizer
        self.n_latent = n_latent
        self.kappa = kappa
        self.alpha = alpha
        self.enc_hidden_dim = enc_hidden_dim
        self.coherence_reg_penalty = coherence_reg_penalty
        self.redundancy_reg_penalty = redundancy_reg_penalty 
        self.batch_size = batch_size
        self.fixed_embedding = fixed_embedding
        self.n_encoding_layers = num_enc_layers
        self.enc_dr = enc_dr
        self.epochs = epochs
        self.vocab_cache = {}        
        self.vocabulary = vocabulary ## nlp.Vocab(nlp.data.count_tokens(vocabulary), unknown_token=None, padding_token=None, bos_token=None, eos_token=None)
        self.ctx = ctx
        self.embedding_source = embedding_source
        self.embedding_size = embedding_size
        self.seed_matrix = seed_matrix
        self.validate_each_epoch = True
        self.wd_freqs = wd_freqs
        self.num_val_words = num_val_words


    def _get_wd_freqs(self, X, max_sample_size=1000000):
        sample_size = min(max_sample_size, X.shape[0])
        data = X[:sample_size] 
        sums = mx.nd.sum(data, axis=0)
        return sums


    def _get_model(self):
        raise NotImplementedError()

    def _npmi(self, X, y, k=10):
        """
        Calculate NPMI(Normalized Pointwise Mutual Information) for data X

        Parameters
        ----------
        X: array-like or sparse matrix, [n_samples, vocab_size]
           Document word matrix.

        k: int, optional (default=10)
           Threshold at which to compute npmi.

        Returns
        -------
        npmi: float
           NPMI score.
        """
        X = mx.nd.sparse.csr_matrix(X)
        sorted_ids = self.model.get_top_k_terms(k)
        num_topics = min(self.n_latent, sorted_ids.shape[-1])
        top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(self.n_latent)]
        npmi_eval = EvaluateNPMI(top_k_words_per_topic)
        npmi = npmi_eval.evaluate_csr_mat(X)
        unique_term_ids = set()
        unique_limit = 5  ## only consider the top 5 terms for each topic when looking at degree of redundancy
        for i in range(num_topics):
            topic_ids = list(top_k_words_per_topic[i][:unique_limit])
            for j in range(len(topic_ids)):
                unique_term_ids.add(topic_ids[j])
        redundancy = (1.0 - (float(len(unique_term_ids)) / num_topics / unique_limit)) ** 2
        return npmi, redundancy
    
        
    def _perplexity(self, dataloader, num_batches, last_batch_size, total_words):
        total_rec_loss = 0
        total_kl_loss  = 0
        for i, (data,labels) in enumerate(dataloader):
            if labels is None:            
                labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
            data = data.as_in_context(self.ctx)
            labels = labels.as_in_context(self.ctx)
            _, kl_loss, rec_loss, _, _, _, log_out = self._forward(self.model, data, labels)
            if i == num_batches - 1:
                total_rec_loss += rec_loss[:last_batch_size].sum().asscalar()
                total_kl_loss  += kl_loss[:last_batch_size].sum().asscalar()
            else:
                total_rec_loss += rec_loss.sum().asscalar()
                total_kl_loss += kl_loss.sum().asscalar()
        if ((total_rec_loss + total_kl_loss) / total_words) < 709.0:
            perplexity = math.exp((total_rec_loss + total_kl_loss) / total_words)
        else:
            perplexity = 1e300
        return perplexity

    def perplexity(self, X, y):
        total_words = X.sum().asscalar()
        dataloader, num_batches, last_batch_size = self._get_val_dataloader(X, y)
        return self._perplexity(dataloader, num_batches, last_batch_size, total_words)

    def _get_val_dataloader(self, val_X, val_y):
        test_size = val_X.shape[0] * val_X.shape[1]
        if test_size < MAX_DESIGN_MATRIX:
            val_X = val_X.tostype('default')
            val_dataloader = DataIterLoader(mx.io.NDArrayIter(val_X, val_y, self.batch_size,
                                                              last_batch_handle='pad', shuffle=False))
        else:
            val_dataloader = DataIterLoader(mx.io.NDArrayIter(val_X, val_y, self.batch_size,
                                                              last_batch_handle='discard', shuffle=False))
        last_batch_size = val_X.shape[0] % self.batch_size
        num_val_batches = val_X.shape[0] // self.batch_size
        if last_batch_size > 0:
            num_val_batches += 1
        return val_dataloader, num_val_batches, last_batch_size

    def validate(self, val_X, val_y):
        process = psutil.Process(os.getpid())
        if self.num_val_words < 0:
            row_cnts = val_X.sum(axis=1)
            sums = row_cnts.sum(axis=0)
            self.num_val_words = sums.asscalar()
        val_dataloader, num_val_batches, last_batch_size = self._get_val_dataloader(val_X, val_y)
        ppl = self._perplexity(val_dataloader, num_val_batches, last_batch_size, self.num_val_words)
        npmi, redundancy = self._npmi(val_X, val_y)
        return ppl, npmi, redundancy


    def fit_with_validation(self, X, y, val_X, val_y):
        wd_freqs = self.wd_freqs if self.wd_freqs is not None else self._get_wd_freqs(X)
        self.model = self._get_model()
        self.model.set_biases(wd_freqs)  ## initialize bias weights to log frequencies
        
        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer, {'learning_rate': self.lr})
        train_dataloader = DataIterLoader(mx.io.NDArrayIter(X, y, self.batch_size, last_batch_handle='discard', shuffle=True))
        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0
        for epoch in range(self.epochs):
            ts_epoch = time.time()
            for i, (data, labels) in enumerate(train_dataloader):
                if labels is None or labels.size == 0:
                    labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
                labels = labels.as_in_context(self.ctx)
                data = data.as_in_context(self.ctx)
                with autograd.record():
                    elbo, kl_loss, rec_loss, entropies, coherence_loss, redundancy_loss, _ = self._forward(self.model, data, labels)
                    elbo_mean = elbo.mean()
                elbo_mean.backward()
                trainer.step(data.shape[0])
            if val_X is not None and (self.validate_each_epoch or epoch == self.epochs-1):
                ppl, npmi, redundancy = self.validate(val_X, val_y)
                if self.reporter:
                    obj = (npmi - redundancy) * self.coherence_coefficient - ( ppl / 1000 )
                    b_obj = max(min(obj, 100.0), -100.0)
                    sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
                    print("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}.".format(epoch, sc_obj, ppl, npmi, redundancy))
                    self.reporter(epoch=epoch+1, objective=sc_obj, time_step=time.time(), coherence=npmi, perplexity=ppl, redundancy=redundancy)
        return sc_obj, npmi, ppl, redundancy
                    
    def fit(self, X, y):
        return self.fit_with_validation(X, y, None, None)


class BowVAE(BaseBowVAE):

    def __init__(self, vocabulary, coherence_coefficient=8.0, reporter=None, num_val_words=-1, wd_freqs=None, ctx=mx.cpu(), lr=0.005, latent_distribution="vmf", optimizer="adam", n_latent=20, kappa=64.0, alpha=1.0, enc_hidden_dim=150, coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0, batch_size=128, embedding_source="random", embedding_size=128, fixed_embedding=False, num_enc_layers=1, enc_dr=0.1, seed_matrix=None, hybridize=False, epochs=40):
        super().__init__(vocabulary, coherence_coefficient, reporter, num_val_words, wd_freqs, ctx, lr, latent_distribution, optimizer, n_latent, kappa, alpha, enc_hidden_dim, coherence_reg_penalty, redundancy_reg_penalty, batch_size, embedding_source, embedding_size, fixed_embedding, num_enc_layers, enc_dr, seed_matrix, hybridize, epochs)


    def npmi(self, X, k=10):
        self._npmi(X, None, k=k)

    def perplexity(self, X):
        """
        Calculate approximate perplexity for data X and y

        Parameters
        ----------
        X: array-like or sparse matrix, [n_samples, vocab_size]
           Document word matrix.

        Returns
        -------
        perplexity: float
           Perplexity score.
        """

        return super().perplexity(X, None)

    def _forward(self, model, data, labels):
        """
        Forward pass of BowVAE model given the supplied data

        Parameters
        ----------
        model: MXNet model that returns elbo, kl_loss, rec_loss, entropies, coherence_loss, redundancy_loss, reconstruction

        data:  {array-like, sparse matrix} of shape (n_train_samples, vocab_size)
           Document word matrix.

        labels: Ignored

        Returns
        -------
        Tuple of elbo, kl_loss, rec_loss, entropies, coherence_loss, redundancy_loss, reconstruction
        """
        return model(data)


    def _get_model(self):
        """
        Returns
        -------
        MXNet model initialized using provided hyperparameters
        """
        #vocab, emb_size = self._initialize_embedding_layer(self.embedding_source, self.embedding_size)
        if self.embedding_source != 'random' and self.vocabulary.embedding is None:
            e_type, e_name = tuple(self.embedding_source.split(':'))
            pt_embedding = nlp.embedding.create(e_type, source=e_name)
            self.vocabulary.set_embedding(pt_embedding)
            emb_size = len(self.vocabulary.embedding.idx_to_vec[0])
        else:
            emb_size = self.embedding_size
        model = \
                BowNTM(self.vocabulary, self.enc_hidden_dim, self.n_latent, emb_size,
                       fixed_embedding=self.fixed_embedding, latent_distrib=self.latent_distrib,
                       coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                       kappa=self.kappa, alpha=self.alpha,
                       batch_size=self.batch_size, n_encoding_layers=self.n_encoding_layers, enc_dr=self.enc_dr,
                       wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, ctx=self.ctx)
        return model
    

    def get_topic_vectors(self):
        """
        Get topic vectors of the fitted model.

        Returns
        -------
        topic_vectors : shape=(n_latent, vocab_size)
            Topic word distribution. topic_distribution[i, j] represents word j in topic i.
        """

        return self.model.get_topic_vectors().asnumpy() 

    def transform(self, X):
        """
        Transform data X according to the fitted model.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape {n_samples, n_features)
            Document word matrix.

        Returns
        -------
        topic_distribution : shape=(n_samples, n_latent)
            Document topic distribution for X
        """

        mx_array = mx.nd.array(X)
        return self.model.encode_data(mx_array).asnumpy()


    def fit(self, X):
        """
        Fit BowVAE model according to the given training data.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_train_samples, vocab_size)
           Document word matrix.

        Returns
        -------
        Tuple of validation/objective scores, if applicable
        """

        return super().fit(X, None)


class MetaBowVAE(BaseBowVAE):

    def __init__(self, vocabulary, coherence_coefficient=8.0, reporter=None, num_val_words=-1, wd_freqs=None, label_map=None, covar_net_layers=1, ctx=mx.cpu(), lr=0.005, latent_distribution="vmf", optimizer="adam", n_latent=20, kappa=64.0, alpha=1.0, enc_hidden_dim=150, coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0, batch_size=128, embedding_source="random", embedding_size=128, fixed_embedding=False, num_enc_layers=1, enc_dr=0.1, seed_matrix=None, hybridize=False, epochs=40):
        super().__init__(vocabulary, coherence_coefficient, reporter, num_val_words, wd_freqs, ctx, lr, latent_distribution, optimizer, n_latent, kappa, alpha, enc_hidden_dim, coherence_reg_penalty, redundancy_reg_penalty, batch_size, embedding_source, embedding_size, fixed_embedding, num_enc_layers, enc_dr, seed_matrix, hybridize, epochs)

        self.covar_net_layers = covar_net_layers
        self.n_covars = len(label_map) if label_map else 1
        self.label_map = label_map

    
    def _get_model(self):
        """
        Returns
        -------
        MXNet model initialized using provided hyperparameters
        """
        if self.embedding_source != 'random' and self.vocabulary.embedding is None:
            e_type, e_name = tuple(self.embedding_source.split(':'))
            pt_embedding = nlp.embedding.create(e_type, source=e_name)
            self.vocabulary.set_embedding(pt_embedding)
            emb_size = len(self.vocabulary.embedding.idx_to_vec[0])
            for word in self.vocabulary.embedding._idx_to_token:
                if (self.vocabulary.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                    self.vocabulary.embedding[word] = mx.nd.random.normal(0, 0.1, emb_size)
        else:
            emb_size = self.embedding_size
        model = \
            MetaDataBowNTM(self.label_map, n_covars=self.n_covars,
                           vocabulary=self.vocabulary, enc_dim=self.enc_hidden_dim, n_latent=self.n_latent, embedding_size=emb_size,
                           fixed_embedding=self.fixed_embedding, latent_distrib=self.latent_distrib,
                           coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                           kappa=self.kappa, alpha=self.alpha,
                           batch_size=self.batch_size, n_encoding_layers=self.n_encoding_layers, enc_dr=self.enc_dr,
                           wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, ctx=self.ctx)
        return model

    
    def _forward(self, model, data, labels):
        """
        Forward pass of BowVAE model given the supplied data

        Parameters
        ----------
        model: MXNet model that returns elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss, redundancy_loss, reconstruction

        data:  {array-like, sparse matrix} of shape (n_train_samples, vocab_size)
           Document word matrix.

        labels: {array-like, sparse matrix} of shape (n_train_samples, n_covars)
           Covariate matrix.

        Returns
        -------
        Tuple of elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss, redundancy_loss, reconstruction
        """

        return model(data, labels)


    def _npmi_per_covariate(self, X, y, k=10):
        """
        Calculate NPMI(Normalized Pointwise Mutual Information) for each covariate for data X

        Parameters
        ----------
        X: array-like or sparse matrix, [n_samples, vocab_size]
           Document word matrix.

        y: array-like or sparse matrix, [n_samples, n_covars]
           Covariate matrix.

        k: int, optional (default=10)
           Threshold at which to compute npmi.

        Returns
        -------
        npmi: Dictionary of npmi scores for each covariate.
        """

        X_train, y_train = X.asnumpy(), y.asnumpy()
        covars = np.unique(y_train, axis=0)
        covar_npmi = {}
        npmi_total = 0
        for covar in covars:
            mask = (y_train == covar).all(axis=1)
            X_covar, y_covar = mx.nd.array(X_train[mask], dtype=np.float32), mx.nd.array(y_train[mask], dtype=np.float32)
            sorted_ids = self.model.get_top_k_terms(X_covar, y_covar)
            top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(self.n_latent)]
            npmi_eval = EvaluateNPMI(top_k_words_per_topic)
            npmi = npmi_eval.evaluate_csr_mat(X_covar)

            if(self.scalar):
                covar_key = covar[0]
            else:
                covar_key = np.where(covar)[0]
            covar_npmi[covar_key] = npmi
            npmi_total += npmi
        return npmi_total / len(covars)

    def _npmi(self, X, y, k=10):
        return super()._npmi(X, y, k)
        #return self._npmi_per_covariate(X, y, k)


    def transform(self, X, y):
        """
        Transform data X and y according to the fitted model.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape {n_samples, n_features)
            Document word matrix.

        y: {array-like, sparse matrix} of shape (n_train_samples, n_covars)
           Covariate matrix.

        Returns
        -------
        topic_distribution : shape=(n_samples, n_latent)
            Document topic distribution for X and y
        """

        x_mxnet, y_mxnet = mx.nd.array(X, dtype=np.float32), mx.nd.array(y, dtype=np.float32)
        return self.model.encode_data_with_covariates(x_mxnet, y_mxnet).asnumpy()


    
