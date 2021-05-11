# coding: utf-8
# Copyright (c) 2020 The MITRE Corporation.
"""
Estimator module to train/fit/estimate individual models with fixed hyperparameters.
Estimators are used by trainers to manage training with specific datasets; in addition,
the estimator API supports inference/encoding with fitted models.
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
import scipy.sparse as sp
import json
from mxnet import autograd
from mxnet import gluon
import gluonnlp as nlp
from pathlib import Path
import umap
#import umap.plot
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, top_k_accuracy_score, roc_auc_score, ndcg_score
from tmnt.data_loading import DataIterLoader, SparseMatrixDataIter
from tmnt.modeling import BowVAEModel, LabeledBowVAEModel, CovariateBowVAEModel, SeqBowVED, DeepAveragingVAEModel
from tmnt.modeling import GeneralizedSDMLLoss, MetricSeqBowVED
from tmnt.eval_npmi import EvaluateNPMI
from tmnt.distribution import HyperSphericalDistribution
import autogluon.core as ag
from itertools import cycle

MAX_DESIGN_MATRIX = 250000000 


class BaseEstimator(object):
    """Base class for all VAE-based estimators.
    
    Parameters:
        log_method (str): Method for logging. 'print' | 'log', optional (default='log')
        quiet (bool): Flag for whether to force minimal logging/ouput. optional (default=False)
        coherence_coefficient (float): Weight to tradeoff influence of coherence vs perplexity in model 
            selection objective (default = 8.0)
        reporter (:class:`autogluon.core.scheduler.Reporter`): Callback reporter to include information for 
            model selection via AutoGluon
        wd_freqs (:class:`mxnet.ndarray.NDArray`): Word frequencies over vocabulary to initialize model biases
        ctx (:class:`mxnet.context`): MXNet context for the estimator
        latent_distribution (str): Latent distribution of the variational autoencoder.
            'logistic_gaussian' | 'vmf' | 'gaussian' | 'gaussian_unitvar', optional (default="vmf")
        optimizer (str): MXNet optimizer (default = "adam")
        lr (float): Learning rate of training. (default=0.005)
        n_latent (int): Size of the latent distribution. optional (default=20)
        kappa (float): Distribution parameter for Von-Mises Fisher distribution, ignored if latent_distribution not 'vmf'. 
            optional (default=64.0)
        alpha (float): Prior parameter for Logistic Gaussian distribution, ignored if latent_distribution not 'logistic_gaussian'. 
            optional (default=1.0)
        coherence_reg_penalty (float): Regularization penalty for topic coherence. optional (default=0.0)
        redundancy_reg_penalty (float): Regularization penalty for topic redundancy. optional (default=0.0)
        batch_size (int): Batch training size. optional (default=128)
        seed_matrix (mxnet matrix): Seed matrix for guided topic model. optional(default=None)
        hybridize (bool): Hybridize underlying mxnet model. optional(default=False)
        epochs (int): Number of training epochs. optional(default=40)
        coherence_via_encoder (bool): Flag to use encoder to derive coherence scores (via gradient attribution)
        pretrained_param_file (str): Path to pre-trained parameter file to initialize weights
        warm_start (bool): Subsequent calls to `fit` will use existing model weights rather than reinitializing
    """

    def __init__(self,
                 log_method='log',
                 quiet=False,
                 coherence_coefficient=8.0,
                 reporter=None,
                 wd_freqs=None,
                 ctx=mx.cpu(),
                 latent_distribution="vmf",
                 optimizer="adam",
                 lr = 0.005, 
                 n_latent=20,
                 kappa=64.0,
                 alpha=1.0,
                 coherence_reg_penalty=0.0,
                 redundancy_reg_penalty=0.0,
                 batch_size=128,
                 seed_matrix=None,
                 hybridize=False,
                 epochs=40,
                 coherence_via_encoder=False,
                 pretrained_param_file=None,
                 warm_start=False):
        self.log_method = log_method
        self.quiet = quiet
        self.model = None
        self.coherence_coefficient = coherence_coefficient
        self.reporter = reporter
        self.wd_freqs = wd_freqs
        self.ctx = ctx
        self.latent_distrib = latent_distribution
        self.optimizer = optimizer
        self.lr = lr
        self.n_latent = n_latent
        self.kappa = kappa
        self.alpha = alpha
        self.coherence_reg_penalty = coherence_reg_penalty
        self.redundancy_reg_penalty = redundancy_reg_penalty
        self.batch_size = batch_size
        self.seed_matrix = seed_matrix
        self.hybridize = hybridize
        self.epochs = epochs
        self.coherence_via_encoder = coherence_via_encoder
        self.pretrained_param_file = pretrained_param_file
        self.warm_start = warm_start
        self.num_val_words = -1 ## will be set later for computing Perplexity on validation dataset
        

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


    def _get_objective_from_validation_result(self, val_result):
        """
        Get the final objective value from the various validation metrics.

        Parameters:
            val_result (dict): Dictionary of validation metrics calculated.        
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
        vocabulary (:class:`gluonnlp.Vocab`): GluonNLP Vocabulary object
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
        pretrained_param_file (str): Path to pre-trained parameter file to initialize weights
        warm_start (bool): Subsequent calls to `fit` will use existing model weights rather than reinitializing
    """
    def __init__(self, vocabulary,
                 validate_each_epoch=False,
                 enc_hidden_dim=150,
                 embedding_source="random",
                 embedding_size=128,
                 fixed_embedding=False,
                 num_enc_layers=1,
                 enc_dr=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc_hidden_dim = enc_hidden_dim
        self.fixed_embedding = fixed_embedding
        self.n_encoding_layers = num_enc_layers
        self.enc_dr = enc_dr
        self.vocabulary = vocabulary 
        self.embedding_source = embedding_source
        self.embedding_size = embedding_size
        self.validate_each_epoch = validate_each_epoch

    @classmethod
    def from_config(cls, config, vocabulary,
                    coherence_via_encoder=False,
                    validate_each_epoch=False, pretrained_param_file=None, wd_freqs=None, reporter=None, ctx=mx.cpu()):
        """
        Create an estimator from a configuration file/object rather than by keyword arguments
        
        Parameters:
            config (str or dict): Path to a json representation of a configuation or TMNT config dictionary
            vocabulary(str or :class:`gluonnlp.Vocab`): Path to a json representation of a vocabulary or GluonNLP vocabulary object
            pretrained_param_file (str): Path to pretrained parameter file if using pretrained model
            wd_freqs (:class:`mxnet.ndarray.NDArray`): Word frequencies over vocabulary to initialize model biases
            reporter (:class:`autogluon.core.scheduler.Reporter`): Callback reporter to include information for model selection via AutoGluon
            ctx (:class:`mxnet.context`): MXNet context for the estimator

        Returns:
            An estimator (:class:`tmnt.estimator.BaseBowEstimator`): Estimator for training and evaluation of a single model
        """
        if isinstance(config, str):
            try:
                with open(config, 'r') as f:
                    config_dict = json.load(f)
            except:
                logging.error("File {} does not appear to be a valid config instance".format(config))
                raise Exception("Invalid Json Configuration File")
            config = ag.space.Dict(**config_dict)
        if isinstance(vocabulary, str):
            try:
                with open(vocabulary, 'r') as f:
                    voc_js = f.read()
            except:
                logging.error("File {} does not appear to be a valid vocabulary file".format(vocabulary))
                raise Exception("Invalid Json Configuration File")            
            vocabulary = nlp.Vocab.from_json(voc_js)
        if vocabulary.embedding is not None:
            emb_size = vocabulary.embedding.idx_to_vec[0].size
        else:
            emb_size = config.embedding.get('size')
            if not emb_size:
                emb_size = config.derived_info.get('embedding_size')
            if not emb_size:
                raise Exception("Embedding size must be provided as the 'size' attribute of 'embedding' or as 'derived_info.embedding_size'")
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
        model = \
                cls(vocabulary, validate_each_epoch=validate_each_epoch,
                    coherence_coefficient=8.0, reporter=reporter, wd_freqs=wd_freqs,
                    ctx=ctx, lr=lr, latent_distribution=latent_distrib, optimizer=optimizer,
                    n_latent=n_latent, kappa=kappa, alpha=alpha, enc_hidden_dim=enc_hidden_dim,
                    coherence_reg_penalty=coherence_reg_penalty,
                    redundancy_reg_penalty=redundancy_reg_penalty, batch_size=batch_size, 
                    embedding_source=embedding_source, embedding_size=emb_size, fixed_embedding=fixed_embedding,
                    num_enc_layers=n_encoding_layers, enc_dr=enc_dr, 
                    epochs=epochs, log_method='log', coherence_via_encoder=coherence_via_encoder,
                    pretrained_param_file = pretrained_param_file,
                    warm_start = (pretrained_param_file is not None))
        return model


    def _get_config(self):
        config = {}
        config['lr']                 = self.lr
        config['enc_hidden_dim']     = self.enc_hidden_dim
        config['n_latent']           = self.n_latent
        config['optimizer']          = self.optimizer
        config['epochs']             = self.epochs
        config['batch_size']         = self.batch_size
        config['num_enc_layers']     = self.n_encoding_layers
        config['enc_dr']             = self.enc_dr
        config['coherence_loss_wt']  = self.coherence_reg_penalty
        config['redundancy_loss_wt'] = self.redundancy_reg_penalty
        config['covar_net_layers']   = 1
        if self.latent_distrib == 'vmf':
            config['latent_distribution'] = {'dist_type':'vmf', 'kappa':self.kappa}
        elif self.latent_distrib == 'logistic_gaussian':
            config['latent_distribution'] = {'dist_type':'logistic_gaussian', 'alpha':self.alpha}
        else:
            config['latent_distribution'] = {'dist_type':'gaussian'}
        if self.embedding_source != 'random':
            config['embedding'] = {'source': self.embedding_source}
        else:
            config['embedding'] = {'source': 'random', 'size': self.embedding_size}
        config['derived_info'] = {'embedding_size': self.embedding_size}
        return config
    

    def write_model(self, model_dir):
        pfile = os.path.join(model_dir, 'model.params')
        sp_file = os.path.join(model_dir, 'model.config')
        vocab_file = os.path.join(model_dir, 'vocab.json')
        logging.info("Model parameters, configuration and vocabulary written to {}".format(model_dir))
        self.model.save_parameters(pfile)
        config = self._get_config()
        specs = json.dumps(config, sort_keys=True, indent=4)
        with io.open(sp_file, 'w') as fp:
            fp.write(specs)
        with io.open(vocab_file, 'w') as fp:
            fp.write(self.model.vocabulary.to_json())


    def _get_wd_freqs(self, X, max_sample_size=1000000):
        sample_size = min(max_sample_size, X.shape[0])
        sums = X.sum(axis=0)
        return sums

    def _get_model(self):
        raise NotImplementedError()

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
                mask = None
            else:
                mask = labels >= 0.0
                mask = mask.as_in_context(self.ctx)
            data = data.as_in_context(self.ctx)
            labels = labels.as_in_context(self.ctx)
            _, kl_loss, rec_loss, _, _, _, _ = self._forward(self.model, data, labels, mask)
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
            val_y = mx.nd.array(val_y) if val_y is not None else None
            val_dataloader = DataIterLoader(mx.io.NDArrayIter(val_X, val_y, test_batch_size,
                                                              last_batch_handle='pad', shuffle=False),
                                            num_batches=num_val_batches, last_batch_size = last_batch_size)
        else:
            val_dataloader = DataIterLoader(SparseMatrixDataIter(val_X, val_y, batch_size = test_batch_size,
                                                                 last_batch_handle='pad', shuffle=False),
                                            )
        return val_dataloader

    def validate(self, val_X, val_y):
        val_dataloader = self._get_val_dataloader(val_X, val_y)
        total_val_words = val_X.sum()
        if self.num_val_words < 0:
            self.num_val_words = total_val_words
        ppl = self._perplexity(val_dataloader, total_val_words)
        if self.coherence_via_encoder:
            npmi, redundancy = self._npmi_with_dataloader(val_dataloader)
        else:
            if val_X.shape[0] > 50000:
                val_X = val_X[:50000]
                val_y = val_y[:50000]
            npmi, redundancy = self._npmi(val_X, val_y)
        return {'ppl': ppl, 'npmi': npmi, 'redundancy': redundancy}
    

    def initialize_with_pretrained(self):
        assert(self.pretrained_param_file is not None)
        self.model = self._get_model()
        self.model.load_parameters(self.pretrained_param_file, allow_missing=False)


    def _get_objective_from_validation_result(self, val_result):
        npmi = val_result['npmi']
        ppl  = val_result['ppl']
        redundancy = val_result['redundancy']
        obj = (npmi - redundancy) * self.coherence_coefficient - ( ppl / 1000 )
        b_obj = max(min(obj, 100.0), -100.0)
        sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
        return sc_obj


    def fit_with_validation(self, X, y, val_X, val_y):
        """
        Fit a model according to the options of this estimator and optionally evaluate on validation data

        Parameters:
            X (tensor): Input training tensor
            y (array): Input labels/co-variates to use (optionally) for co-variate models
            val_X (tensor): Validateion input tensor
            val_y (array): Validation co-variates

        Returns:
            (tuple): Tuple of:
               sc_obj, npmi, perplexity, redundancy
        """
        wd_freqs = self.wd_freqs if self.wd_freqs is not None else self._get_wd_freqs(X)
        x_size = X.shape[0] * X.shape[1]
        if x_size > MAX_DESIGN_MATRIX:
            logging.info("Sparse matrix has total size = {}. Using Sparse Matrix data batcher.".format(x_size))
            train_dataloader = \
                DataIterLoader(SparseMatrixDataIter(X, y, batch_size = self.batch_size, last_batch_handle='discard', shuffle=True))
        else:
            y = mx.nd.array(y) if y is not None else None
            X = mx.nd.sparse.csr_matrix(X)
            train_dataloader = DataIterLoader(mx.io.NDArrayIter(X, y, self.batch_size, last_batch_handle='discard', shuffle=True))

        if self.model is None or not self.warm_start:
            self.model = self._get_model()
            self.model.set_biases(mx.nd.array(wd_freqs).squeeze())  ## initialize bias weights to log frequencies
        
        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer, {'learning_rate': self.lr})
        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0
        v_res = None
        for epoch in range(self.epochs):
            ts_epoch = time.time()
            elbo_losses = []
            lab_losses  = []
            for i, (data, labels) in enumerate(train_dataloader):
                if labels is None:
                    labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
                    mask = None
                else:
                    if len(labels.shape) > 1:
                        mask = labels.sum(axis=1) >= 0.0
                    else:
                        mask = labels >= 0.0
                    mask = mask.as_in_context(self.ctx)
                labels = labels.as_in_context(self.ctx)
                data = data.as_in_context(self.ctx)
                with autograd.record():
                    elbo, kl_loss, rec_loss, entropies, coherence_loss, redundancy_loss, lab_loss = \
                        self._forward(self.model, data, labels, mask=mask)
                    elbo_mean = elbo.mean()
                elbo_mean.backward()
                trainer.step(1)
                if not self.quiet:
                    elbo_losses.append(float(elbo_mean.asscalar()))
                    if lab_loss is not None:
                        lab_losses.append(float(lab_loss.mean().asscalar()))
            if not self.quiet and not self.validate_each_epoch:
                elbo_mean = np.mean(elbo_losses) if len(elbo_losses) > 0 else 0.0
                lab_mean  = np.mean(lab_losses) if len(lab_losses) > 0 else 0.0
                self._output_status("Epoch [{}] finished in {} seconds. [elbo = {}, label loss = {}]"
                                    .format(epoch+1, (time.time()-ts_epoch), elbo_mean, lab_mean))
            mx.nd.waitall()
            if val_X is not None and (self.validate_each_epoch or epoch == self.epochs-1):
                v_res = self.validate(val_X, val_y)
                sc_obj = self._get_objective_from_validation_result(v_res)
                self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                                        .format(epoch+1, sc_obj, v_res['ppl'], v_res['npmi'], v_res['redundancy']))
                if self.reporter:
                    self.reporter(epoch=epoch+1, objective=sc_obj, time_step=time.time(),
                                  coherence=v_res['npmi'], perplexity=v_res['ppl'], redundancy=v_res['redundancy'])
        mx.nd.waitall()
        return sc_obj, v_res

                    
    def fit(self, X, y):
        self.fit_with_validation(X, y, None, None)
        return self


class BowEstimator(BaseBowEstimator):

    def __init__(self, vocabulary, *args, **kwargs):
        super().__init__(vocabulary, *args, **kwargs)

    @classmethod
    def from_config(cls, *args, **kwargs):
        return super().from_config(*args, **kwargs)

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

    def _forward(self, model, data, labels, mask):
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
                BowVAEModel(self.enc_hidden_dim, emb_size, n_encoding_layers=self.n_encoding_layers,
                            enc_dr=self.enc_dr, fixed_embedding=self.fixed_embedding, vocabulary=self.vocabulary, n_latent=self.n_latent, 
                            latent_distrib=self.latent_distrib, 
                            coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                       kappa=self.kappa, alpha=self.alpha,
                       batch_size=self.batch_size, 
                       wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, n_covars=0, ctx=self.ctx)
        if self.pretrained_param_file is not None:
            model.load_parameters(self.pretrained_param_file, allow_missing=False)
        return model
    

    def get_topic_vectors(self):
        """
        Get topic vectors of the fitted model.

        Returns:
            topic_vectors (:class:`NDArray`): Topic word distribution. 
                topic_distribution[i, j] represents word j in topic i. shape=(n_latent, vocab_size)
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

    def fit(self, X):
        """
        Fit BowVAE model according to the given training data.

        Parameters:
            X ({array-like, sparse matrix}): Document word matrix of shape (n_train_samples, vocab_size)

        Returns:
            (`BowVAE`): Self
        """
        return super().fit(X, None)


class LabeledBowEstimator(BaseBowEstimator):

    def __init__(self, vocabulary, *args, n_labels=0,  gamma=1.0, multilabel=False, **kwargs):
        super().__init__(vocabulary, *args, **kwargs)
        self.multilabel = multilabel
        self.gamma = gamma
        self.n_labels = n_labels

    @classmethod
    def from_config(cls, n_labels, gamma, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        est.n_labels = n_labels
        est.gamma = gamma
        return est

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
            LabeledBowVAEModel(n_labels=self.n_labels, gamma=self.gamma, multilabel=self.multilabel,
                           vocabulary=self.vocabulary, enc_dim=self.enc_hidden_dim, n_latent=self.n_latent, embedding_size=emb_size,
                           fixed_embedding=self.fixed_embedding, latent_distrib=self.latent_distrib,
                           coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                           kappa=self.kappa, alpha=self.alpha,
                           batch_size=self.batch_size, n_encoding_layers=self.n_encoding_layers, enc_dr=self.enc_dr,
                           wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, ctx=self.ctx)
        return model


    def _get_config(self):
        config = super()._get_config()
        config['n_labels'] = self.n_labels
        return config
    

    def _forward(self, model, data, labels, mask):
        self.train_data = data
        self.train_labels = labels
        return model(data, labels, mask)


    def _get_objective_from_validation_result(self, v_res):
        topic_obj = super()._get_objective_from_validation_result(v_res)
        acc = v_res['accuracy']
        logging.info("topic obj = {}, acc = {}".format(topic_obj, acc))
        return topic_obj + self.gamma * acc


    def validate(self, val_X, val_y):
        v_res = super().validate(val_X, val_y)
        val_loader = self._get_val_dataloader(val_X, val_y)
        tot_correct = 0
        tot = 0
        bs = min(val_X.shape[0], self.batch_size)
        num_std_batches = val_X.shape[0] // bs
        last_batch_size = val_X.shape[0] % bs
        for i, (data, labels) in enumerate(val_loader):
            if i < num_std_batches - 1:
                data = data.as_in_context(self.ctx)
                labels = labels.as_in_context(self.ctx)
                predictions = self.model.predict(data)
                correct = mx.nd.argmax(predictions, axis=1) == labels
                tot_correct += mx.nd.sum(correct).asscalar()
                tot += (data.shape[0] - (labels < 0.0).sum().asscalar()) # subtract off labels < 0
        acc = float(tot_correct) / float(tot)
        v_res['accuracy'] = acc
        print("v_res = {}".format(v_res))
        return v_res
    

    def get_topic_vectors(self):
        """
        Get topic vectors of the fitted model.

        Returns:
            topic_vectors (:class:`NDArray`): Topic word distribution. topic_distribution[i, j] represents word j in topic i. 
                shape=(n_latent, vocab_size)
        """

        return self.model.get_topic_vectors(self.train_data, self.train_labels)


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
    

class CovariateBowEstimator(BaseBowEstimator):

    def __init__(self, vocabulary, n_covars=0, *args, **kwargs):
        
        super().__init__(vocabulary, *args, **kwargs)

        self.covar_net_layers = 1 ### XXX - temp hardcoded
        self.n_covars = n_covars


    @classmethod
    def from_config(cls, n_covars, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        est.n_covars = n_covars
        print("Estimator from_config with type = {}".format(type(est)))
        print("Number of covars for estimator = {}".format(est.n_covars))
        return est
    
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
            CovariateBowVAEModel(n_covars=self.n_covars,
                           vocabulary=self.vocabulary, enc_dim=self.enc_hidden_dim, n_latent=self.n_latent, embedding_size=emb_size,
                           fixed_embedding=self.fixed_embedding, latent_distrib=self.latent_distrib,
                           coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                           kappa=self.kappa, alpha=self.alpha,
                           batch_size=self.batch_size, n_encoding_layers=self.n_encoding_layers, enc_dr=self.enc_dr,
                           wd_freqs=self.wd_freqs, seed_mat=self.seed_matrix, ctx=self.ctx)
        return model


    def _get_config(self):
        config = super()._get_config()
        config['n_covars'] = self.n_covars
        return config
    
    
    def _forward(self, model, data, labels, mask):
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

            #if(self.label_map):
            #    covar_key = covar[0]
            #else:
            #    covar_key = np.where(covar)[0][0]
            covar_keky = covar[0]
            covar_npmi[covar_key] = npmi
            npmi_total += npmi
        return npmi_total / len(covars)

    def _npmi(self, X, y, k=10):
        return super()._npmi(X, y, k)
        #return self._npmi_per_covariate(X, y, k)

    def _get_objective_from_validation_result(self, v_res):
        return v_res['npmi']

    def validate(self, X, y):
        npmi, redundancy = self._npmi(X, y)
        return {'npmi': npmi, 'redundancy': redundancy, 'ppl': 0.0}

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

    def __init__(self, bert_base, *args,
                 bert_model_name = 'bert_12_768_12',
                 bert_data_name = 'book_corpus_wiki_en_uncased',
                 bow_vocab = None,
                 n_labels = 0,
                 log_interval=5,
                 warmup_ratio=0.1,
                 gamma=1.0,
                 multilabel=False,
                 decoder_lr = 0.01,
                 checkpoint_dir = None,
                 optimizer = 'bertadam',
                 classifier_dropout = 0.0,
                 **kwargs):
        super(SeqBowEstimator, self).__init__(*args, optimizer=optimizer, **kwargs)
        self.minimum_lr = 1e-9
        self.checkpoint_dir = checkpoint_dir
        self.bert_base = bert_base
        self.bert_model_name = bert_model_name
        self.bert_data_name = bert_data_name
        self.has_classifier = n_labels >= 2
        self.classifier_dropout = classifier_dropout
        self.multilabel = multilabel
        self.n_labels = n_labels
        self.metric = mx.metric.Accuracy()
        self.warmup_ratio = warmup_ratio
        self.log_interval = log_interval
        self.loss_function = gluon.loss.SigmoidBCELoss() if multilabel else gluon.loss.SoftmaxCELoss()
        self.gamma = gamma
        self.decoder_lr = decoder_lr
        self._bow_matrix = None
        self.bow_vocab = bow_vocab


    @classmethod
    def from_config(cls, config, reporter=None, log_interval=1, pretrained_param_file=None, ctx=mx.cpu()):
        if isinstance(config, str):
            try:
                with open(config, 'r') as f:
                    config_dict = json.load(f)
            except:
                logging.error("File {} does not appear to be a valid config instance".format(config))
                raise Exception("Invalid Json Configuration File")
            config = ag.space.Dict(**config_dict)
        lr = config.lr
        decoder_lr = config.decoder_lr
        optimizer = config.optimizer
        n_latent = int(config.n_latent)
        batch_size = int(config.batch_size)
        epochs = int(config.epochs)
        ldist_def = config.latent_distribution
        kappa = 0.0
        alpha = 1.0
        latent_distrib = ldist_def.dist_type
        #embedding_source = config.embedding_source
        redundancy_reg_penalty = config.redundancy_reg_penalty
        warmup_ratio = config.warmup_ratio
        n_labels = config.n_labels
        if latent_distrib == 'vmf':
            kappa = ldist_def.kappa
        elif latent_distrib == 'logistic_gaussian':
            alpha = ldist_def.alpha
        bert_base, _ = nlp.model.get_model(config.bert_model_name,  
                                           dataset_name=config.bert_data_name,
                                           pretrained=True, ctx=ctx, use_pooler=True,
                                           use_decoder=False, use_classifier=False)
        model = cls(bert_base,
                    n_labels=n_labels,
                    coherence_coefficient=8.0,
                    reporter=reporter,
                    latent_distribution=latent_distrib,
                    n_latent=n_latent,
                    redundancy_reg_penalty=redundancy_reg_penalty,
                    kappa = kappa,
                    alpha=alpha,
                    batch_size=batch_size,
                    warmup_ratio = warmup_ratio,
                    optimizer = optimizer,
                    epochs = epochs,
                    lr = lr,
                    decoder_lr = decoder_lr,
                    pretrained_param_file = pretrained_param_file,
                    warm_start = (pretrained_param_file is not None),
                    ctx=ctx,
                    log_interval=log_interval)
        return model
    

    def initialize_with_pretrained(self):
        assert(self.pretrained_param_file is not None)
        self.model = self._get_model()
        self.model.load_parameters(self.pretrained_param_file, allow_missing=False)


    def _get_model_bias_initialize(self, train_data):
        model = self._get_model()
        tr_bow_counts = self._get_bow_wd_counts(train_data)
        model.initialize_bias_terms(tr_bow_counts)
        return model
        
    
    def _get_model(self):
        latent_dist = HyperSphericalDistribution(self.n_latent, kappa=64.0, ctx=self.ctx)
        model = SeqBowVED(self.bert_base, latent_dist, num_classes=self.n_labels, n_latent=self.n_latent,
                          bow_vocab_size = len(self.bow_vocab), dropout=self.classifier_dropout)
        model.decoder.initialize(init=mx.init.Xavier(), ctx=self.ctx)
        model.latent_dist.initialize(init=mx.init.Xavier(), ctx=self.ctx)
        model.latent_dist.post_init(self.ctx)
        if model.has_classifier:
            model.classifier.initialize(init=mx.init.Normal(0.02), ctx=self.ctx)
        return model
    

    def _get_config(self):
        config = {}
        config['lr'] = self.lr
        config['decoder_lr'] = self.decoder_lr
        config['optimizer'] = self.optimizer
        config['n_latent'] = self.n_latent
        config['n_labels'] = self.n_labels
        config['batch_size'] = self.batch_size
        if self.latent_distrib == 'vmf':
            config['latent_distribution'] = {'dist_type':'vmf', 'kappa':self.kappa}
        elif self.latent_distrib == 'logistic_gaussian':
            config['latent_distribution'] = {'dist_type':'logistic_gaussian', 'alpha':self.alpha}
        else:
            config['latent_distribution'] = {'dist_type':'gaussian'}
        config['epochs'] = self.epochs
        #config['embedding_source'] = self.embedding_source
        config['redundancy_reg_penalty'] = self.redundancy_reg_penalty
        config['warmup_ratio'] = self.warmup_ratio
        config['bert_model_name'] = self.bert_model_name
        config['bert_data_name'] = self.bert_data_name
        config['classifier_dropout'] = self.classifier_dropout
        return config

    def write_model(self, model_dir, suf=''):
        pfile = os.path.join(model_dir, ('model.params' + suf))
        conf_file = os.path.join(model_dir, ('model.config' + suf))
        vocab_file = os.path.join(model_dir, ('vocab.json' + suf))
        self.model.save_parameters(pfile)
        config = self._get_config()
        specs = json.dumps(config, sort_keys=True, indent=4)
        with open(conf_file, 'w') as f:
            f.write(specs)
        with open(vocab_file, 'w') as f:
            f.write(self.bow_vocab.to_json())


    def log_train(self, batch_id, batch_num, metric, step_loss, rec_loss, red_loss, class_loss,
                  log_interval, epoch_id, learning_rate):
        """Generate and print out the log message for training. """
        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]
        self._output_status("Epoch {} Batch {}/{} loss={}, (rec_loss = {}), (red_loss = {}), (class_loss = {}) lr={:.10f}, metrics: {:.10f}"
              .format(epoch_id+1, batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval, red_loss/log_interval,
                      class_loss/log_interval, learning_rate, *metric_val))

    def log_eval(self, batch_id, batch_num, metric, step_loss, rec_loss, log_interval):
        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]
        self._output_status("Batch {}/{} loss={} (rec_loss = {}), metrics: {:.10f}"
              .format(batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval, *metric_val))

    def _get_bow_matrix(self, dataloader, cache=False):
        bow_matrix = []
        max_rows = 2000000000 / len(self.bow_vocab)
        logging.info("Maximum rows for BOW matrix = {}".format(max_rows))
        rows = 0
        for i, seqs in enumerate(dataloader):
            bow_batch = list(seqs[3].squeeze(axis=1))
            rows += len(bow_batch)
            if i >= max_rows:
                break
            bow_matrix.extend(bow_batch)
        bow_matrix = mx.nd.stack(*bow_matrix)
        if cache:
            self._bow_matrix = bow_matrix
        return bow_matrix

    def _get_bow_wd_counts(self, dataloader):
        sums = mx.nd.zeros(len(self.bow_vocab))
        for i, seqs in enumerate(dataloader):
            bow_batch = seqs[3].squeeze(axis=1)
            sums += bow_batch.sum(axis=0)
        return sums

    def _get_objective_from_validation_result(self, val_result):
        npmi = val_result['npmi']
        ppl  = val_result['ppl']
        redundancy = val_result['redundancy']
        obj = (npmi - redundancy) * self.coherence_coefficient - ( ppl / 1000 )
        b_obj = max(min(obj, 100.0), -100.0)
        sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
        return sc_obj

    def _get_losses(self, model, batch_data):
        input_ids, valid_length, type_ids, bow, label = batch_data
        elbo_ls, rec_ls, kl_ls, red_ls, out = model(
            input_ids.as_in_context(self.ctx), type_ids.as_in_context(self.ctx),
            valid_length.astype('float32').as_in_context(self.ctx), bow.as_in_context(self.ctx))
        if self.has_classifier:
            label = label.as_in_context(self.ctx)
            label_ls = self.loss_function(out, label)
            label_ls = label_ls.mean()
            total_ls = (self.gamma * label_ls) + elbo_ls.mean()
            ## update label metric (e.g. accuracy)
            self.metric.update(labels=[label], preds=[out])
        else:
            total_ls = elbo_ls.mean()
            label_ls = mx.nd.zeros(total_ls.shape)        
        return elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls

    def _get_unlabeled_losses(self, model, batch_data):
        inputs, vl, tt, bow, _ = batch_data
        elbo_ls, rec_ls, kl_ls, red_ls, out = model(
            inputs.as_in_context(self.ctx), tt.as_in_context(self.ctx),
            vl.astype('float32').as_in_context(self.ctx), bow.as_in_context(self.ctx))
        total_ls = elbo_ls.mean() / self.gamma
        return elbo_ls, rec_ls, kl_ls, red_ls, total_ls
        

    def fit_with_validation(self, train_data, dev_data, num_train_examples, aux_data=None):
        """Training function."""
        if self.model is None or not self.warm_start:
            model = self._get_model_bias_initialize(train_data)
            self.model = model
        
        accumulate = False

        all_model_params = model.collect_params()
        optimizer_params = {'learning_rate': self.lr, 'epsilon': 1e-6, 'wd': 0.02}
        non_decoder_params = {**model.bert.collect_params()}
        decoder_params     = {**model.decoder.collect_params(), **model.latent_dist.collect_params()}
        if self.has_classifier:
            decoder_params.update(model.classifier.collect_params())

        trainer = gluon.Trainer(non_decoder_params, self.optimizer,
                                    optimizer_params, update_on_kvstore=False)
        dec_trainer = gluon.Trainer(decoder_params, 'adam', {'learning_rate': self.decoder_lr, 'epsilon': 1e-6, 'wd': 0.00001})
        #if args.dtype == 'float16':
        #    amp.init_trainer(trainer)

        step_size = self.batch_size * accumulate if accumulate else self.batch_size
        num_train_steps = int(num_train_examples / step_size * self.epochs) + 1
        warmup_ratio = self.warmup_ratio
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        step_num = 0

        # Do not apply weight decay on LayerNorm and bias terms
        for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0
        # Collect differentiable parameters
        params = [p for p in all_model_params.values() if p.grad_req != 'null']
        clipped_params = []
        for p in non_decoder_params.values():
            if p.grad_req != 'null':
                clipped_params.append(p)
        
        # Set grad_req if gradient accumulation is required
        if (accumulate and accumulate > 1) or (aux_data is not None):
            for p in params:
                p.grad_req = 'add'

        loss_details = { 'step_loss': 0.0, 'elbo_loss': 0.0, 'red_loss': 0.0, 'class_loss': 0.0 }
        def update_loss_details(total_ls, elbo_ls, red_ls, class_ls):
            loss_details['step_loss'] += total_ls.mean().asscalar()
            loss_details['elbo_loss'] += elbo_ls.mean().asscalar()
            loss_details['red_loss'] += red_ls.mean().asscalar()
            if class_ls is not None:
                loss_details['class_loss'] += class_ls.mean().asscalar()
            
        for epoch_id in range(self.epochs):
            self.metric.reset()
            all_model_params.zero_grad()
            aux_data = [None] if aux_data is None else aux_data
            paired_dataloaders = zip(train_data, cycle(aux_data)) if len(train_data) > len(aux_data) else zip(cycle(train_data), aux_data)
            #for (batch_id, seqs) in enumerate(train_data):
            for (batch_id, (seqs, aux_seqs)) in enumerate(paired_dataloaders):
                # learning rate schedule
                if step_num < num_warmup_steps:
                    new_lr = self.lr * (step_num+1) / num_warmup_steps
                else:
                    non_warmup_steps = step_num - num_warmup_steps
                    offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                    new_lr = self.lr - offset * self.lr
                new_lr = max(new_lr, self.minimum_lr)
                trainer.set_learning_rate(new_lr)
                # forward and backward with optional auxilliary data
                with mx.autograd.record():
                    elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls = self._get_losses(model, seqs)
                total_ls.backward()
                if aux_seqs is not None:
                    with mx.autograd.record():
                        elbo_ls_2, rec_ls_2, kl_ls_2, red_ls_2, total_ls_2 = self._get_unlabeled_losses(model, aux_seqs)
                    total_ls_2.backward()

                update_loss_details(total_ls, elbo_ls, red_ls, label_ls)
                if aux_seqs is not None:
                    update_loss_details(total_ls_2, elbo_ls_2, red_ls_2, None)

                # update
                if not accumulate or (batch_id + 1) % accumulate == 0:
                    trainer.allreduce_grads()
                    dec_trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(clipped_params, 1.0, check_isfinite=True)
                    trainer.update(accumulate if accumulate else 1)
                    dec_trainer.update(accumulate if accumulate else 1)
                    step_num += 1
                    if (accumulate and accumulate > 1) or (aux_data is not None or (len(aux_data) > 1)):
                        # set grad to zero for gradient accumulation
                        all_model_params.zero_grad()
                if (batch_id + 1) % (self.log_interval) == 0:
                    self.log_train(batch_id, len(train_data), self.metric, loss_details['step_loss'],
                                   loss_details['elbo_loss'], loss_details['red_loss'], loss_details['class_loss'], self.log_interval,
                                   epoch_id, trainer.learning_rate)
                    ## reset loss details
                    for d in loss_details:
                        loss_details[d] = 0.0
            mx.nd.waitall()

            # inference on dev data
            if dev_data is not None:
                sc_obj, v_res = self._perform_validation(model, dev_data, epoch_id)
            else:
                sc_obj, v_res = None, None
            if self.checkpoint_dir:
                self.write_model(self.checkpoint_dir, suf=str(epoch_id))
        return sc_obj, v_res


    def _compute_coherence(self, model, k, test_data, log_terms=False):
        num_topics = model.n_latent
        sorted_ids = model.get_top_k_terms(k, ctx=self.ctx)
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
        #if log_terms:
        #    top_k_tokens = [list(map(lambda x: self.vocabulary.idx_to_token[x], list(li))) for li in top_k_words_per_topic]
        #    for i in range(num_topics):
        #        logging.info("Topic {}: {}".format(i, top_k_tokens[i]))
        return npmi, redundancy
    

    def _perform_validation(self, model, dev_data, epoch_id):
        v_res, metric_nm, metric_val = self.validate(model, dev_data)
        sc_obj = self._get_objective_from_validation_result(v_res)
        if 'accuracy' in v_res:
            self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}. Accuracy = {}."
                                .format(epoch_id, sc_obj, v_res['ppl'], v_res['npmi'], v_res['redundancy'], v_res['accuracy']))
        else:
            self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                                .format(epoch_id, sc_obj, v_res['ppl'], v_res['npmi'], v_res['redundancy']))
        if self.reporter:
            if 'accuracy' in v_res:
                self.reporter(epoch=epoch_id+1, objective=sc_obj, time_step=time.time(), coherence=v_res['npmi'],
                              perplexity=v_res['ppl'], redundancy=v_res['redundancy'], accuracy=v_res['accuracy'])
            else:
                self.reporter(epoch=epoch_id+1, objective=sc_obj, time_step=time.time(), coherence=v_res['npmi'],
                              perplexity=v_res['ppl'], redundancy=v_res['redundancy'])
        return sc_obj, v_res
                
    
    def validate(self, model, dataloader):
        bow_matrix = self._bow_matrix if self._bow_matrix is not None else self._get_bow_matrix(dataloader, cache=True)
        num_words = bow_matrix.sum().asscalar()
        npmi, redundancy = self._compute_coherence(model, 10, bow_matrix, log_terms=True)
        self.metric.reset()
        step_loss = 0
        elbo_loss  = 0
        total_rec_loss = 0.0
        total_kl_loss  = 0.0
        for batch_id, seqs in enumerate(dataloader):
            elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls = self._get_losses(model, seqs)
            total_rec_loss += rec_ls.sum().asscalar()
            total_kl_loss  += kl_ls.sum().asscalar()
            step_loss += total_ls.mean().asscalar()
            elbo_loss  += elbo_ls.mean().asscalar()
            if (batch_id + 1) % (self.log_interval) == 0:
                logging.info('All loss terms: {}, {}, {}, {}, {}, {}'.format(elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls))
                self.log_eval(batch_id, len(dataloader), self.metric, step_loss, elbo_loss, self.log_interval)
                step_loss = 0
                elbo_loss = 0
        likelihood = (total_rec_loss + total_kl_loss) / num_words
        if likelihood < 709.0:
            perplexity = math.exp(likelihood)
        else:
            perplexity = 1e300
        v_res = {'ppl':perplexity, 'npmi': npmi, 'redundancy': redundancy}
        metric_nm = 0.0
        metric_val = 0.0
        if self.has_classifier:
            metric_nm, metric_val = self.metric.get()
            if not isinstance(metric_nm, list):
                metric_nm, metric_val = [metric_nm], [metric_val]
            self._output_status("Validation metric: {:.6}".format(metric_val[0]))
            v_res['accuracy'] = metric_val[0]
        return v_res, metric_nm, metric_val



class SeqBowMetricEstimator(SeqBowEstimator):

    def __init__(self, *args, sdml_smoothing_factor=0.3, fixed_data=None, fixed_test_data=None, plot_dir=None, **kwargs):
        super(SeqBowMetricEstimator, self).__init__(*args, **kwargs)
        self.loss_function = GeneralizedSDMLLoss(smoothing_parameter=sdml_smoothing_factor)
        self.fixed_batch = None
        self.fixed_test_batch = None
        self.plot_dir = plot_dir
        if fixed_data:
            self.fixed_batch = next(enumerate(fixed_data))[1] # take the first batch and fix
            if fixed_test_data:
                self.fixed_test_batch = next(enumerate(fixed_test_data))[1]


    @classmethod
    def from_config(cls, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        return est
        
    def _get_model(self, bow_size=-1):
        bow_size = bow_size if bow_size > 1 else len(self.bow_vocab)
        latent_dist = HyperSphericalDistribution(self.n_latent, kappa=64.0, ctx=self.ctx)
        model = MetricSeqBowVED(self.bert_base, latent_dist, n_latent=self.n_latent,
                                bow_vocab_size = len(self.bow_vocab), dropout=self.classifier_dropout)
        model.decoder.initialize(init=mx.init.Xavier(), ctx=self.ctx)
        model.latent_dist.initialize(init=mx.init.Xavier(), ctx=self.ctx)
        model.latent_dist.post_init(self.ctx)
        if self.pretrained_param_file is not None:
            model.load_parameters(self.pretrained_param_file, allow_missing=False)
        return model

    def _get_model_bias_initialize(self, train_data):
        model = self._get_model()
        tr_bow_matrix = self._get_bow_matrix(train_data)
        model.initialize_bias_terms(tr_bow_matrix.sum(axis=0))
        return model
        

    def _get_bow_matrix(self, dataloader, cache=False):
        bow_matrix = []
        for _, seqs in enumerate(dataloader):
            if self.fixed_batch:
                batch_1 = seqs
            else:
                batch_1, batch_2 = seqs                
                bow_matrix.extend(list(batch_2[3].squeeze(axis=1)))
            bow_matrix.extend(list(batch_1[3].squeeze(axis=1)))
        if self.fixed_batch:
            bow_matrix.extend(list(self.fixed_batch[3].squeeze(axis=1)))
        bow_matrix = mx.nd.stack(*bow_matrix)
        if cache:
            self._bow_matrix = bow_matrix
        return bow_matrix

    def _ff_batch(self, model, batch_data, on_test=False):
        if on_test:
            if self.fixed_test_batch:
                batch1 = batch_data
                batch2 = self.fixed_test_batch
            else:
                batch1, batch2 = batch_data
        else:
            if self.fixed_batch:
                batch1 = batch_data
                batch2 = self.fixed_batch
            else:
                batch1, batch2 = batch_data
        in1, vl1, tt1, bow1, label1 = batch1
        in2, vl2, tt2, bow2, label2 = batch2
        elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2 = model(
            in1.as_in_context(self.ctx), tt1.as_in_context(self.ctx),
            vl1.astype('float32').as_in_context(self.ctx), bow1.as_in_context(self.ctx),
            in2.as_in_context(self.ctx), tt2.as_in_context(self.ctx),
            vl2.astype('float32').as_in_context(self.ctx), bow2.as_in_context(self.ctx))
        return elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2

    def _get_losses(self, model, batch_data):
        elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2 = self._ff_batch(model, batch_data)
        label1 = label1.as_in_context(self.ctx)
        label2 = label2.as_in_context(self.ctx)
        label_ls = self.loss_function(z_mu1, label1, z_mu2, label2)
        label_ls = label_ls.mean()
        total_ls = (self.gamma * label_ls) + elbo_ls.mean()
        return elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls

    def _get_unlabeled_losses(self, model, batch_data):
        in1, vl1, tt1, bow1, label1 = batch_data
        elbo_ls, rec_ls, kl_ls, red_ls = model.unpaired_input_forward(
            in1.as_in_context(self.ctx), tt1.as_in_context(self.ctx),
            vl1.astype('float32').as_in_context(self.ctx), bow1.as_in_context(self.ctx))
        total_ls = elbo_ls / self.gamma
        return elbo_ls, rec_ls, kl_ls, red_ls, total_ls
    
    def classifier_validate(self, model, dataloader, epoch_id):
        posteriors = []
        ground_truth = []
        ground_truth_idx = []
        emb2 = None
        emb1 = []
        for batch_id, seqs in enumerate(dataloader):
            elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2 = self._ff_batch(model, seqs, on_test=True)
            label_mat = self.loss_function._compute_labels(mx.ndarray, label1, label2)
            dists = self.loss_function._compute_distances(z_mu1, z_mu2)
            probs = mx.nd.softmax(-dists, axis=1).asnumpy()
            posteriors += list(probs)
            label1 = np.array(label1.squeeze().asnumpy(), dtype='int')
            ground_truth_idx += list(label1) ## index values for labels
            gt = np.zeros((label1.shape[0], int(mx.nd.max(label2).asscalar())+1))
            gt[np.arange(label1.shape[0]), label1] = 1
            ground_truth += list(gt)
            if emb2 is None:
                emb2 = z_mu2.asnumpy()
            emb1 += list(z_mu1.asnumpy())
        posteriors = np.array(posteriors)
        ground_truth = np.array(ground_truth)
        ground_truth_idx = np.array(ground_truth_idx)
        avg_prec = average_precision_score(ground_truth, posteriors, average='weighted')
        logging.info('EVALUTAION: Ground truth indices: {}'.format(list(ground_truth_idx)))
        try:
            auroc = roc_auc_score(ground_truth, posteriors, average='weighted')
        except:
            auroc = 0.0
            logging.error('ROC computation failed')
        ndcg = ndcg_score(ground_truth, posteriors)
        top_acc_1 = top_k_accuracy_score(ground_truth_idx, posteriors, k=1)        
        top_acc_2 = top_k_accuracy_score(ground_truth_idx, posteriors, k=2)
        top_acc_3 = top_k_accuracy_score(ground_truth_idx, posteriors, k=3)
        top_acc_4 = top_k_accuracy_score(ground_truth_idx, posteriors, k=4)
        y = np.where(ground_truth > 0)[1]
        if self.plot_dir:
            ofile = self.plot_dir + '/' + 'plot_' + str(epoch_id) + '.png'
            umap_model = umap.UMAP(n_neighbors=4, min_dist=0.5, metric='euclidean')
            embeddings = umap_model.fit_transform(np.array(emb1))
            #mapper = umap_model.fit(np.array(emb1))
            plt.scatter(*embeddings.T, c=y, s=0.8, alpha=0.9, cmap='coolwarm')
            #umap.plot.points(mapper, labels=y)
            plt.savefig(ofile)
            plt.close("all")
        return {'avg_prec': avg_prec, 'top_1': top_acc_1, 'top_2': top_acc_2, 'top_3': top_acc_3, 'top_4': top_acc_4,
                'au_roc': auroc, 'ndcg': ndcg}

            
    def _perform_validation(self, model, dev_data, epoch_id):
        v_res = self.classifier_validate(model, dev_data, epoch_id)
        self._output_status("Epoch [{}]. Objective = {} ==> Avg. Precision = {}, AuROC = {}, NDCG = {} [acc@1= {}, acc@2={}, acc@3={}, acc@4={}]"
                            .format(epoch_id, v_res['avg_prec'], v_res['avg_prec'], v_res['au_roc'], v_res['ndcg'],
                                    v_res['top_1'], v_res['top_2'], v_res['top_3'], v_res['top_4']))
        if self.reporter:
            self.reporter(epoch=epoch_id+1, objective=v_res['avg_prec'], time_step=time.time(), coherence=0.0,
                          perplexity=0.0, redundancy=0.0)
        return v_res['avg_prec'], v_res


class DeepAveragingBowEstimator(BaseEstimator):

    def __init__(self, vocabulary, n_labels, gamma, emb_dim, emb_dr, seq_length, *args, **kwargs):
        super(DeepAveragingBowEstimator, self).__init__(*args, **kwargs)
        self.vocabulary = vocabulary
        self.n_labels = n_labels
        self.gamma = gamma
        self.emb_in_dim = len(vocabulary)
        self.emb_out_dim = emb_dim
        self.emb_dr = emb_dr
        self.seq_length = seq_length
        self.validate_each_epoch = False

    def _get_model(self):
        model = DeepAveragingVAEModel(self.n_labels, self.gamma, self.emb_in_dim, self.emb_out_dim , self.emb_dr, self.seq_length,
                                      vocabulary=self.vocabulary, n_latent=self.n_latent, latent_distrib=self.latent_distrib,
                                      batch_size=self.batch_size, wd_freqs=self.wd_freqs, ctx=self.ctx)
        return model
              

    def _forward(self, model, ids, lens, bow, labels, l_mask):
        return model(ids, lens, bow, labels, l_mask)


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

        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer, {'learning_rate': self.lr})
        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0
        
        for epoch in range(self.epochs):
            ts_epoch = time.time()
            elbo_losses = []
            lab_losses  = []
            for i, seqs in enumerate(dataloader):
                ids, valid_len, output_bow, labels = seqs
                if labels is None:
                    labels = mx.nd.expand_dims(mx.nd.zeros(), 1)
                    mask = None
                else:
                    if len(labels.shape) > 1:
                        mask = labels.sum(axis=1) >= 0.0
                    else:
                        mask = labels >= 0.0
                    mask = mask.as_in_context(self.ctx)
                ids    = ids.as_in_context(self.ctx)
                labels = labels.as_in_context(self.ctx)
                valid_len = valid_len.as_in_context(self.ctx)
                output_bow = output_bow.as_in_context(self.ctx)

                with autograd.record():
                    elbo, kl_loss, rec_loss, entropies, coherence_loss, redundancy_loss, lab_loss = \
                        self._forward(self.model, ids, valid_len, output_bow, labels, mask)
                    elbo_mean = elbo.mean()
                elbo_mean.backward()
                trainer.step(1)
                if not self.quiet:
                    elbo_losses.append(float(elbo_mean.asscalar()))
                    if lab_loss is not None:
                        lab_losses.append(float(lab_loss.mean().asscalar()))
            if not self.quiet and not self.validate_each_epoch:
                elbo_mean = np.mean(elbo_losses) if len(elbo_losses) > 0 else 0.0
                lab_mean  = np.mean(lab_losses) if len(lab_losses) > 0 else 0.0
                self._output_status("Epoch [{}] finished in {} seconds. [elbo = {}, label loss = {}]"
                                    .format(epoch+1, (time.time()-ts_epoch), elbo_mean, lab_mean))
            if val_X is not None and (self.validate_each_epoch or epoch == self.epochs-1):
              _, val_X_sp = val_X
              npmi, redundancy = self._npmi(val_X_sp, None)
              self._output_status("NPMI ==> {}".format(npmi))
                #ppl, npmi, redundancy = self.validate(val_X, val_y)
                #if self.reporter:
                #    obj = (npmi - redundancy) * self.coherence_coefficient - ( ppl / 1000 )
                #    b_obj = max(min(obj, 100.0), -100.0)
                #    sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
                #    self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                #                        .format(epoch+1, sc_obj, ppl, npmi, redundancy))
                #    self.reporter(epoch=epoch+1, objective=sc_obj, time_step=time.time(),
                #                  coherence=npmi, perplexity=ppl, redundancy=redundancy)
              
        return sc_obj, {'npmi': npmi, 'ppl': ppl, 'redundancy': redundancy}

    
