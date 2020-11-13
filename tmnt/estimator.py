# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import logging
import math
import logging
import time
import io
import os
import psutil
import mxnet as mx
import numpy as np

from mxnet import autograd
from mxnet import gluon
import gluonnlp as nlp
from pathlib import Path

from tmnt.data_loading import DataIterLoader, SparseMatrixDataIter
from tmnt.modeling import BowVAEModel, MetaDataBowVAEModel, BertBowVED
from tmnt.eval_npmi import EvaluateNPMI

MAX_DESIGN_MATRIX = 250000000 


class BaseEstimator(object):

    def __init__(self, log_method='log', quiet=False):
        self.log_method = log_method
        self.quiet = quiet
        self.model = None

    def _output_status(self, status_string):
        if self.log_method == 'print':
            print(status_string)
        elif self.log_method == 'log':
            logging.info(status_string)

    def get_topic_vectors(self):
        raise NotImplementedError()


    def _get_model(self):
        """
        Returns:
            (:class:`mxnet.gluon.HybridBlock`): MXNet model initialized using provided hyperparameters
        """

        raise NotImplementedError()


    def fit(self, X, y):
        """
        Fit VAE model according to the given training data X with optional co-variates y.
  
        Parameters:
            X (tensor): representing input data
            y (tensor): representing covariate/labels associated with data elements
        """
        raise NotImplementedError()
    

    def fit_with_validation(self, X, y, val_X, val_Y):
        """
        Fit VAE model according to the given training data X with optional co-variates y;
        validate (potentially each epoch) with validation data val_X and optional co-variates val_Y
  
        Parameters:
            X (tensor): representing training data
            y (tensor): representing covariate/labels associated with data elements in training data
            val_X (tensor): representing validation data
            val_y (tensor): representing covariate/labels associated with data elements in validation data
        """
        raise NotImplementedError()


class BaseBowEstimator(BaseEstimator):
    """
    Bag of words variational autoencoder algorithm

    Parameters:
        vocabulary (list[string]): Vocabulary list used for pretrained emeddings
        lr (float): Learning rate of training. (default=0.005)
        latent_distribution (str): Latent distribution of the variational autoencoder.
            'logistic_gaussian' | 'vmf' | 'gaussian' | 'gaussian_unitvar', optional (default="vmf")
        n_latent (int): Size of the latent distribution. optional (default=20)
        kappa (float): Distribution parameter for Von-Mises Fisher distribution, ignored if latent_distribution not 'vmf'. 
            optional (default=64.0)
        alpha (float): Prior parameter for Logistic Gaussian distribution, ignored if latent_distribution not 'logistic_gaussian'. 
            optional (default=1.0)
        enc_hidden_dim (int): Size of hidden encoder layers. optional (default=150)
        coherence_reg_penalty (float): Regularization penalty for topic coherence. optional (default=0.0)
        redundancy_reg_penalty (float): Regularization penalty for topic redundancy. optional (default=0.0)
        batch_size (int): Batch training size. optional (default=128)
        embedding_source (str): Word embedding source for vocabulary.
            'random' | 'glove' | 'fasttext' | 'word2vec', optional (default='random')
        embedding_size (int): Word embedding size, ignored if embedding_source not 'random'. optional (default=128)
        fixed_embedding (bool): Enable fixed embeddings. optional(default=False)
        num_enc_layers (int): Number of layers in encoder. optional(default=1)
        enc_dr (float): Dropout probability in encoder. optional(default=0.1)
        seed_matrix (mxnet matrix): Seed matrix for guided topic model. optional(default=None)
        hybridize (bool): Hybridize underlying mxnet model. optional(default=False)
        epochs (int): Number of training epochs. optional(default=40)
        log_method (str): Method for logging. 'print' | 'log', optional (default='log')
        quiet (bool): Flag for whether to force minimal logging/ouput. optional (default=False)
        coherence_via_encoder (bool): Flag 
    """
    def __init__(self, vocabulary, coherence_coefficient=8.0, reporter=None, num_val_words=-1, wd_freqs=None, ctx=mx.cpu(), lr=0.005, latent_distribution="vmf", optimizer="adam", n_latent=20, kappa=64.0, alpha=1.0, enc_hidden_dim=150, coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0, batch_size=128, embedding_source="random", embedding_size=128, fixed_embedding=False, num_enc_layers=1, enc_dr=0.1, seed_matrix=None, hybridize=False, epochs=40, log_method='print', quiet=False, coherence_via_encoder=False):
        
        super().__init__(log_method=log_method, quiet=quiet)
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
        self.model = None
        self.coherence_via_encoder = coherence_via_encoder

    def _get_wd_freqs(self, X, max_sample_size=1000000):
        sample_size = min(max_sample_size, X.shape[0])
        data = X[:sample_size] 
        sums = data.sum(axis=0)
        return sums

    def _get_model(self):
        raise NotImplementedError()


    def _npmi(self, X, y, k=10):
        """
        Calculate NPMI(Normalized Pointwise Mutual Information) for data X

        Parameters:
            X (array-like or sparse matrix): Document word matrix. shape [n_samples, vocab_size]
            k (int): Threshold at which to compute npmi. optional (default=10)

        Returns:
            npmi (float): NPMI score.
        """
        sorted_ids = self.model.get_ordered_terms()
        num_topics = min(self.n_latent, sorted_ids.shape[-1])
        top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t])] for t in range(self.n_latent)]
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

    def _npmi_with_dataloader(self, dataloader, k=10):
        sorted_ids = self.model.get_ordered_terms_encoder(dataloader) if self.coherence_via_encoder else self.model.get_ordered_terms()
        num_topics = min(self.n_latent, sorted_ids.shape[-1])
        top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t])] for t in range(self.n_latent)]
        npmi_eval = EvaluateNPMI(top_k_words_per_topic)
        npmi = npmi_eval.evaluate_csr_loader(dataloader)
        unique_term_ids = set()
        unique_limit = 5  ## only consider the top 5 terms for each topic when looking at degree of redundancy
        for i in range(num_topics):
            topic_ids = list(top_k_words_per_topic[i][:unique_limit])
            for j in range(len(topic_ids)):
                unique_term_ids.add(topic_ids[j])
        redundancy = (1.0 - (float(len(unique_term_ids)) / num_topics / unique_limit)) ** 2
        return npmi, redundancy
    
    def _perplexity(self, dataloader, total_words):
        total_rec_loss = 0
        total_kl_loss  = 0
        last_batch_size = dataloader.last_batch_size
        num_batches = dataloader.num_batches
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
        dataloader = self._get_val_dataloader(X, y)
        self.num_val_words = X.sum()
        return self._perplexity(dataloader, self.num_val_words)

    def _get_val_dataloader(self, val_X, val_y):
        test_size = val_X.shape[0] * val_X.shape[1]
        test_batch_size = min(val_X.shape[0], self.batch_size)
        last_batch_size = val_X.shape[0] % test_batch_size if test_batch_size < val_X.shape[0] else test_batch_size
        num_val_batches = val_X.shape[0] // test_batch_size
        if last_batch_size > 0 and last_batch_size < test_batch_size:
            num_val_batches += 1
        if test_size < MAX_DESIGN_MATRIX:
            val_X = mx.nd.sparse.csr_matrix(val_X).tostype('default')
            val_dataloader = DataIterLoader(mx.io.NDArrayIter(val_X, val_y, test_batch_size,
                                                              last_batch_handle='pad', shuffle=False),
                                            num_batches=num_val_batches, last_batch_size = last_batch_size)
        elif test_size < 1000000000:
            val_X = mx.nd.sparse.csr_matrix(val_X)
            val_dataloader = DataIterLoader(mx.io.NDArrayIter(val_X, val_y, test_batch_size,
                                                              last_batch_handle='pad', shuffle=False),
                                                num_batches=num_val_batches, last_batch_size = last_batch_size)
        else:
            val_dataloader = DataIterLoader(SparseMatrixDataIter(val_X, val_y, batch_size = test_batch_size,
                                                                     last_batch_handle='pad', shuffle=False))
        return val_dataloader

    def validate(self, val_X, val_y):
        val_dataloader = self._get_val_dataloader(val_X, val_y)
        ppl = self._perplexity(val_dataloader, self.num_val_words)
        if self.coherence_via_encoder:
            npmi, redundancy = self._npmi_with_dataloader(val_dataloader)
        else:
            if val_X.shape[0] > 50000:
                val_X = val_X[:50000]
                val_y = val_y[:50000]
            npmi, redundancy = self._npmi(val_X, val_y)
        return ppl, npmi, redundancy


    def fit_with_validation(self, X, y, val_X, val_y):
        wd_freqs = self.wd_freqs if self.wd_freqs is not None else self._get_wd_freqs(X)
        val_y = mx.nd.array(val_y) if val_y is not None else None
        y = mx.nd.array(y) if y is not None else None
        x_size = X.shape[0] * X.shape[1]
        if x_size > MAX_DESIGN_MATRIX:
            logging.info("Sparse matrix has total size = {}. Using Sparse Matrix data batcher.".format(x_size))
            train_dataloader = \
                DataIterLoader(SparseMatrixDataIter(X, y, batch_size = self.batch_size, last_batch_handle='discard', shuffle=True))
        else:
            X = mx.nd.sparse.csr_matrix(X)
            train_dataloader = DataIterLoader(mx.io.NDArrayIter(X, y, self.batch_size, last_batch_handle='discard', shuffle=True))
        self.model = self._get_model()
        self.model.set_biases(mx.nd.array(wd_freqs).squeeze())  ## initialize bias weights to log frequencies
        
        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer, {'learning_rate': self.lr})
        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0
        for epoch in range(self.epochs):
            ts_epoch = time.time()
            for i, (data, labels) in enumerate(train_dataloader):
                if labels is None:
                    labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
                labels = labels.as_in_context(self.ctx)
                data = data.as_in_context(self.ctx)
                with autograd.record():
                    elbo, kl_loss, rec_loss, entropies, coherence_loss, redundancy_loss, _ = \
                        self._forward(self.model, data, labels)
                    elbo_mean = elbo.mean()
                elbo_mean.backward()
                trainer.step(data.shape[0])
            if not self.quiet and not self.validate_each_epoch:
                self._output_status("Epoch [{}] finished in {} seconds. ".format(epoch+1, (time.time()-ts_epoch)))
            if val_X is not None and (self.validate_each_epoch or epoch == self.epochs-1):
                ppl, npmi, redundancy = self.validate(val_X, val_y)
                if self.reporter:
                    obj = (npmi - redundancy) * self.coherence_coefficient - ( ppl / 1000 )
                    b_obj = max(min(obj, 100.0), -100.0)
                    sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
                    self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                                        .format(epoch+1, sc_obj, ppl, npmi, redundancy))
                    self.reporter(epoch=epoch+1, objective=sc_obj, time_step=time.time(), coherence=npmi, perplexity=ppl, redundancy=redundancy)
        return sc_obj, npmi, ppl, redundancy

                    
    def fit(self, X, y):
        self.fit_with_validation(X, y, None, None)
        return self



class BowEstimator(BaseBowEstimator):

    def __init__(self, vocabulary, coherence_coefficient=8.0, reporter=None, num_val_words=-1, wd_freqs=None, ctx=mx.cpu(), lr=0.005, latent_distribution="vmf", optimizer="adam", n_latent=20, kappa=64.0, alpha=1.0, enc_hidden_dim=150, coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0, batch_size=128, embedding_source="random", embedding_size=128, fixed_embedding=False, num_enc_layers=1, enc_dr=0.1, seed_matrix=None, hybridize=False, epochs=40, log_method='print', quiet=False, coherence_via_encoder=False, pretrained_param_file=None):

        super().__init__(vocabulary, coherence_coefficient, reporter, num_val_words, wd_freqs, ctx, lr, latent_distribution, optimizer, n_latent, kappa, alpha, enc_hidden_dim, coherence_reg_penalty, redundancy_reg_penalty, batch_size, embedding_source, embedding_size, fixed_embedding, num_enc_layers, enc_dr, seed_matrix, hybridize, epochs, log_method, quiet, coherence_via_encoder)
        self.pretrained_param_file = pretrained_param_file


    def npmi(self, X, k=10):
        return self._npmi(X, None, k=k)

    def perplexity(self, X):
        """
        Calculate approximate perplexity for data X and y

        Parameters:
            X (array-like or sparse matrix): Document word matrix of shape [n_samples, vocab_size]

        Returns:
           (float): Perplexity score.
        """

        return super().perplexity(X, None)

    def _forward(self, model, data, labels):
        """
        Forward pass of BowVAE model given the supplied data

        Parameters:
            model (:class:`BowVAEModel`): Core VAE model for bag-of-words topic model
            data (:class:`mxnet.ndarray.NDArray`): Document word matrix of shape (n_train_samples, vocab_size)
            labels: Ignored

        Returns:
            (tuple): Tuple of:
                elbo, kl_loss, rec_loss, entropies, coherence_loss, redundancy_loss, reconstruction
        """
        return model(data)


    def _get_model(self):
        """
        Initializes embedding weights and returns a `BowVAEModel` with hyperparameters provided.

        Returns:
            (:class:`BowVAEModel`) initialized using provided hyperparameters
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
                BowVAEModel(self.vocabulary, self.enc_hidden_dim, self.n_latent, emb_size,
                       fixed_embedding=self.fixed_embedding, latent_distrib=self.latent_distrib,
                       coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                       kappa=self.kappa, alpha=self.alpha,
                       batch_size=self.batch_size, n_encoding_layers=self.n_encoding_layers, enc_dr=self.enc_dr,
                       wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, ctx=self.ctx)
        if self.pretrained_param_file is not None:
            model.load_parameters(self.pretrained_param_file, allow_missing=False)
        return model
    

    def get_topic_vectors(self):
        """
        Get topic vectors of the fitted model.

        Returns:
            topic_vectors (:class:`NDArray`): Topic word distribution. topic_distribution[i, j] represents word j in topic i. shape=(n_latent, vocab_size)
        """

        return self.model.get_topic_vectors() 

    def transform(self, X):
        """
        Transform data X according to the fitted model.

        Parameters:
            X ({array-like, sparse matrix}): Document word matrix of shape {n_samples, n_features}

        Returns:
            (:class:`mxnet.ndarray.NDArray`) topic_distribution: shape=(n_samples, n_latent) Document topic distribution for X
        """

        mx_array = mx.nd.array(X,dtype='float32')
        return self.model.encode_data(mx_array).asnumpy()


    ### Example using fit API:
    ### from tmnt.models.bow.bow_doc_loader import load_vocab
    ### from tmnt.models.bow.bow_vae import BowVAE
    ### from sklearn.datasets import load_svmlight_file
    ### vocab = load_vocab('/Users/wellner/Devel/tmnt-data/20news/raw/20news.vocab')
    ### X,y = load_svmlight_file('/Users/wellner/Devel/tmnt-data/20news/raw/train.vec', n_features=len(vocab))
    ### vae = BowVAE(vocab)
    ### estimator = vae.fit(X)
    ### encodings = estimator.transform(X)

    def fit(self, X):
        """
        Fit BowVAE model according to the given training data.

        Parameters:
            X ({array-like, sparse matrix}): Document word matrix of shape (n_train_samples, vocab_size)

        Returns:
            (`BowVAE`): Self
        """

        return super().fit(X, None)



class MetaBowEstimator(BaseBowEstimator):

    def __init__(self, vocabulary, coherence_coefficient=8.0, reporter=None, num_val_words=-1, wd_freqs=None, label_map=None, covar_net_layers=1, ctx=mx.cpu(), lr=0.005, latent_distribution="vmf", optimizer="adam", n_latent=20, kappa=64.0, alpha=1.0, enc_hidden_dim=150, coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0, batch_size=128, embedding_source="random", embedding_size=128, fixed_embedding=False, num_enc_layers=1, enc_dr=0.1, seed_matrix=None, hybridize=False, epochs=40, log_method='print'):
        
        super().__init__(vocabulary, coherence_coefficient, reporter, num_val_words, wd_freqs, ctx, lr, latent_distribution, optimizer, n_latent, kappa, alpha, enc_hidden_dim, coherence_reg_penalty, redundancy_reg_penalty, batch_size, embedding_source, embedding_size, fixed_embedding, num_enc_layers, enc_dr, seed_matrix, hybridize, epochs, log_method)

        self.covar_net_layers = covar_net_layers
        self.n_covars = len(label_map) if label_map else 1
        self.label_map = label_map

    
    def _get_model(self):
        """
        Returns
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
            MetaDataBowVAEModel(self.label_map, n_covars=self.n_covars,
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

        Parameters:
            model (MXNet model): Model that returns elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss, redundancy_loss, reconstruction
            data ({array-like, sparse matrix}): Document word matrix of shape (n_train_samples, vocab_size) 
            labels ({array-like, sparse matrix}): Covariate matrix of shape (n_train_samples, n_covars)

        Returns:
            (tuple): Tuple of: 
                elbo, kl_loss, rec_loss, l1_pen, entropies, coherence_loss, redundancy_loss, reconstruction
        """
        self.train_data = data
        self.train_labels = labels
        return model(data, labels)


    def _npmi_per_covariate(self, X, y, k=10):
        """
        Calculate NPMI(Normalized Pointwise Mutual Information) for each covariate for data X

        Parameters:
            X (array-like or sparse matrix): Document word matrix. shape [n_samples, vocab_size]
            y (array-like or sparse matrix): Covariate matrix. shape [n_samples, n_covars]
            k (int): Threshold at which to compute npmi. optional (default=10)

        Returns:
            (dict): Dictionary of npmi scores for each covariate.
        """

        X_train = X.toarray()
        y_train = y
        covars = np.unique(y_train, axis=0)
        covar_npmi = {}
        npmi_total = 0
        for covar in covars:
            mask = (y_train == covar).all(axis=1)
            X_covar, y_covar = mx.nd.array(X_train[mask], dtype=np.float32), mx.nd.array(y_train[mask], dtype=np.float32)
            sorted_ids = self.model.get_ordered_terms_with_covar_at_data(X_covar,k, y_covar)
            top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(self.n_latent)]
            npmi_eval = EvaluateNPMI(top_k_words_per_topic)
            npmi = npmi_eval.evaluate_csr_mat(X_covar)

            if(self.label_map):
                covar_key = covar[0]
            else:
                covar_key = np.where(covar)[0][0]
            covar_npmi[covar_key] = npmi
            npmi_total += npmi
        return npmi_total / len(covars)

    def _npmi(self, X, y, k=10):
        return super()._npmi(X, y, k)
        #return self._npmi_per_covariate(X, y, k)

    def get_topic_vectors(self):
        """
        Get topic vectors of the fitted model.

        Returns:
            topic_vectors (:class:`NDArray`): Topic word distribution. topic_distribution[i, j] represents word j in topic i. 
                shape=(n_latent, vocab_size)
        """

        return self.model.get_topic_vectors(self.train_data, self.train_labels)

    def transform(self, X, y):
        """
        Transform data X and y according to the fitted model.

        Parameters:
            X ({array-like, sparse matrix}): Document word matrix of shape {n_samples, n_features)
            y ({array-like, sparse matrix}): Covariate matrix of shape (n_train_samples, n_covars)

        Returns:
            ({array-like, sparse matrix}): Document topic distribution for X and y of shape=(n_samples, n_latent)
        """

        x_mxnet, y_mxnet = mx.nd.array(X, dtype=np.float32), mx.nd.array(y, dtype=np.float32)
        return self.model.encode_data_with_covariates(x_mxnet, y_mxnet).asnumpy()
    
    

class SeqBowEstimator(BaseEstimator):

    def __init__(self, bert_base, vocab, coherence_coefficient=8.0, reporter=None, latent_distribution="vmf", n_latent=20, redundancy_reg_penalty=0.0, kappa=64.0, batch_size=32, kld=1.0, wd_freqs=None, num_val_words=-1, warmup_ratio=0.1, optimizer="adam", epochs=3, gen_lr=0.000001, dec_lr=0.01, min_lr=0.00000005, ctx=mx.cpu(), log_interval=1, log_method='log'):
        super().__init__(log_method=log_method)
        self.bert_base = bert_base
        self.coherence_coefficient = coherence_coefficient
        self.reporter = reporter
        self.vocabulary = vocab
        self.latent_distribution = latent_distribution
        self.log_interval = log_interval
        self.redundancy_reg_penalty = redundancy_reg_penalty
        self.n_latent = n_latent
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
                           n_latent=self.n_latent, 
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
        top_k_words_per_topic = [[ int(i) for i in list(sorted_ids[:k, t])] for t in range(num_topics)]
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
        """
        Estimate a topic model by fine-tuning pre-trained BERT encoder.
        
        Parameters:
            X ({array-like, sparse matrix}): Training document word matrix of shape {n_samples, n_features)
            y ({array-like, sparse matrix}): Training covariate matrix of shape (n_train_samples, n_covars)
            val_X ({array-like, sparse matrix}): Validation document word matrix of shape {n_samples, n_features)
            val_y ({array-like, sparse matrix}): Validation covariate matrix of shape (n_train_samples, n_covars)
        """
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
    
