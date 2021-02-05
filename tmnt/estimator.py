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
import json
from mxnet import autograd
from mxnet import gluon
import gluonnlp as nlp
from pathlib import Path

from tmnt.data_loading import DataIterLoader, SparseMatrixDataIter
from tmnt.modeling import BowVAEModel, LabeledBowVAEModel, CovariateBowVAEModel, BertBowVED, LabeledBertBowVED, DeepAveragingVAEModel
from tmnt.eval_npmi import EvaluateNPMI
import autogluon.core as ag

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
        self.validate_each_epoch = False

    @classmethod
    def from_config(cls, config, vocabulary, pretrained_param_file=None, wd_freqs=None, reporter=None, ctx=mx.cpu()):
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
                cls(vocabulary, coherence_coefficient=8.0, reporter=reporter, wd_freqs=wd_freqs,
                             ctx=ctx, lr=lr, latent_distribution=latent_distrib, optimizer=optimizer,
                             n_latent=n_latent, kappa=kappa, alpha=alpha, enc_hidden_dim=enc_hidden_dim,
                             coherence_reg_penalty=coherence_reg_penalty,
                             redundancy_reg_penalty=redundancy_reg_penalty, batch_size=batch_size, 
                             embedding_source=embedding_source, embedding_size=emb_size, fixed_embedding=fixed_embedding,
                             num_enc_layers=n_encoding_layers, enc_dr=enc_dr, 
                             epochs=epochs, log_method='log', coherence_via_encoder=False,
                             pretrained_param_file = pretrained_param_file)
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
            if val_X is not None and (self.validate_each_epoch or epoch == self.epochs-1):
                v_res = self.validate(val_X, val_y)
                sc_obj = self._get_objective_from_validation_result(v_res)
                if self.reporter:
                    self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                                        .format(epoch+1, sc_obj, v_res['ppl'], v_res['npmi'], v_res['redundancy']))
                    self.reporter(epoch=epoch+1, objective=sc_obj, time_step=time.time(),
                                  coherence=v_res['npmi'], perplexity=v_res['ppl'], redundancy=v_res['redundancy'])
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
        return {'npmi': npmi, 'redundancy': redundancy}


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

    def __init__(self, bert_base, bert_vocab, vocab, bow_embedding_source='random',
                 warmup_ratio=0.1, gen_lr=0.000001, dec_lr=0.01, min_lr=0.00000005, log_interval=1, max_batches=-1, epochs=3, *args, **kwargs):
        super(SeqBowEstimator, self).__init__(epochs=epochs, *args, **kwargs)
        self.log_interval = log_interval
        self.bert_base = bert_base
        self.embedding_source = bow_embedding_source
        self.vocabulary = vocab
        self.bert_vocab = bert_vocab
        self.gen_lr = gen_lr
        self.dec_lr = dec_lr
        self.min_lr = min_lr
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.offset_factor = 1.0
        self.validate_each_epoch = True
        self.max_batches = max_batches

    @classmethod
    def from_config(cls, config, vocab, reporter, wd_freqs, log_interval=1, ctx=mx.cpu()):
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
        warmup_ratio = config.warmup_ratio
        if latent_distrib == 'vmf':
            kappa = ldist_def.kappa
        elif latent_distrib == 'logistic_gaussian':
            alpha = ldist_def.alpha
        bert_base, bert_vocab = nlp.model.get_model('bert_12_768_12',  
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
        model = cls(bert_base, bert_vocab, vocab,
                    bow_embedding_source=embedding_source,
                          coherence_coefficient=8.0,
                          reporter=reporter,
                          latent_distribution=latent_distrib,
                          n_latent=n_latent,
                          redundancy_reg_penalty=redundancy_reg_penalty,
                          kappa = kappa,
                    alpha=alpha,
                          batch_size=batch_size,
                          kld=1.0, wd_freqs=wd_freqs, 
                          warmup_ratio = warmup_ratio,
                          optimizer = optimizer,
                          epochs = epochs,
                          gen_lr = gen_lr,
                          dec_lr = dec_lr,
                          min_lr = min_lr,
                          ctx=ctx,
                          log_interval=log_interval)
        return model

    def _get_model(self):
        model = BertBowVED(self.bert_base, self.vocabulary, self.latent_distrib, 
                           n_latent=self.n_latent, 
                           kappa = self.kappa,
                           alpha = self.alpha,
                           batch_size = self.batch_size,
                           kld=1.0, wd_freqs=self.wd_freqs,
                           redundancy_reg_penalty=self.redundancy_reg_penalty,
                           ctx=self.ctx)
        return model


    def _get_config(self):
        config = {}
        config['gen_lr'] = self.gen_lr
        config['min_lr'] = self.min_lr
        config['dec_lr'] = self.dec_lr
        config['optimizer'] = self.optimizer
        config['n_latent'] = self.n_latent
        config['batch_size'] = self.batch_size
        if self.latent_distrib == 'vmf':
            config['latent_distribution'] = {'dist_type':'vmf', 'kappa':self.kappa}
        elif self.latent_distrib == 'logistic_gaussian':
            config['latent_distribution'] = {'dist_type':'logistic_gaussian', 'alpha':self.alpha}
        else:
            config['latent_distribution'] = {'dist_type':'gaussian'}
        config['epochs'] = self.epochs
        config['embedding_source'] = self.embedding_source
        config['redundancy_reg_penalty'] = self.redundancy_reg_penalty
        config['warmup_ratio'] = self.warmup_ratio
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
            f.write(self.vocabulary.to_json())


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
            input_ids, valid_length, type_ids, output_vocab, labels = seqs
            _, rec_loss, kl_loss, _, _, _ = self.model(input_ids.as_in_context(self.ctx), type_ids.as_in_context(self.ctx),
                                                    valid_length.astype('float32').as_in_context(self.ctx),
                                                    output_vocab.as_in_context(self.ctx),
                                                    labels.as_in_context(self.ctx))
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


    def _get_objective_from_validation_result(self, val_result):
        npmi = val_result['npmi']
        ppl  = val_result['ppl']
        redundancy = val_result['redundancy']
        obj = (npmi - redundancy) * self.coherence_coefficient - ( ppl / 1000 )
        b_obj = max(min(obj, 100.0), -100.0)
        sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
        return sc_obj


    def validate(self, model, bow_train, bow_val_X, val_y, dataloader):
        npmi, redundancy = self._compute_coherence(model, 10, bow_train, log_terms=True)
        last_batch_size = bow_val_X.shape[0] % self.batch_size
        if last_batch_size > 0:
            num_batches = (bow_val_X.shape[0] // self.batch_size) + 1
        else:
            num_batches = bow_val_X.shape[0] // self.batch_size
            last_batch_size = self.batch_size
        num_val_words = bow_val_X.sum()
        ppl = self._perplexity(dataloader, num_val_words, last_batch_size, num_batches)
        return {'ppl': ppl, 'npmi': npmi, 'redundancy': redundancy}
    


    def fit_with_validation(self, X, y, val_X, val_y):
        """
        Estimate a topic model by fine-tuning pre-trained BERT encoder.
        
        Parameters:
            X ({array-like, sparse matrix}): Training document word matrix of shape {n_samples, n_features)
            y ({array-like, sparse matrix}): Training covariate matrix of shape (n_train_samples, n_covars)
            val_X ({array-like, sparse matrix}): Validation document word matrix of shape {n_samples, n_features)
            val_y ({array-like, sparse matrix}): Validation covariate matrix of shape (n_train_samples, n_covars)
        Returns:
            (tuple): Tuple containing:
                - estimator (:class:`tmnt.estimator.BaseEstimator`) VAE model estimator
                - objective (float): Resulting objective value
                - result_details (dict): Estimator-specific metrics on validation data
        """
        seq_train, bow_train = X
        model = self._get_model()
        self.model = model
        ## merge y values into the dataset for X
        if y is not None:
            seq_train._data.append(mx.nd.array(y))
        else:
            seq_train._data.append(mx.nd.zeros(bow_train.shape[0]))
        dataloader = mx.gluon.data.DataLoader(seq_train, batch_size=self.batch_size,
                                              shuffle=True, last_batch='rollover')
        if val_X is not None:
            seq_val, bow_val = val_X
            if val_y is not None:
                seq_val._data.append(mx.nd.array(val_y))
            else:
                seq_val._data.append(mx.nd.zeros(bow_val.shape[0]))
            dataloader_val = mx.gluon.data.DataLoader(seq_val, batch_size=self.batch_size, last_batch='rollover',
                                                       shuffle=False)

        num_train_examples = len(seq_train)
        num_train_steps = int(num_train_examples / self.batch_size * self.epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        step_num = 0
        differentiable_params = []

        lr = self.gen_lr

        bert_params = model.encoder.collect_params()
        if isinstance(model, LabeledBertBowVED):
            bert_params.update(model.lab_decoder.collect_params())
            
        gen_trainer = gluon.Trainer(bert_params, self.optimizer,
                                    {'learning_rate': self.gen_lr, 'epsilon': 1e-6, 'wd':self.weight_decay})
        var_params = {**model.latent_dist.collect_params(), **model.decoder.collect_params()}
        dec_trainer = gluon.Trainer(var_params, 'adam', {'learning_rate': self.dec_lr, 'epsilon': 1e-6})

        # Do not apply weight decay on LayerNorm and bias terms
        all_model_params = model.collect_params()
        
        for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

        params = [p for p in all_model_params.values() if p.grad_req != 'null']

        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0
        v_res = None
        for epoch_id in range(self.epochs):
            all_model_params.zero_grad()
            
            step_loss = 0
            step_recon_ls = 0
            step_kl_ls = 0
            step_red_ls = 0
            step_lab_ls = 0
            if self.max_batches > 0 and step_num >= self.max_batches:
                break
            for batch_id, seqs in enumerate(dataloader):
                step_num += 1
                if step_num < num_warmup_steps:
                    new_lr = lr * step_num / num_warmup_steps
                else:
                    offset = (step_num - num_warmup_steps) * lr / ((num_train_steps - num_warmup_steps) * self.offset_factor)
                    new_lr = max(lr - offset, self.min_lr)
                gen_trainer.set_learning_rate(new_lr)
                with mx.autograd.record():
                    input_ids, valid_length, type_ids, output_vocab, labels = seqs
                    ls, recon_ls, kl_ls, redundancy_ls, predictions, lab_loss = \
                        model(input_ids.as_in_context(self.ctx), type_ids.as_in_context(self.ctx),
                              valid_length.astype('float32').as_in_context(self.ctx),
                              output_vocab.as_in_context(self.ctx), labels.as_in_context(self.ctx))
                    ls = ls.mean()
                    ls.backward()
                nlp.utils.clip_grad_global_norm(params, 1)
                dec_trainer.update(1)
                gen_trainer.update(1) # step of 1 since we averaged loss over batch
                step_loss += ls.asscalar()
                step_recon_ls += recon_ls.mean().asscalar()
                step_kl_ls += kl_ls.mean().asscalar()
                step_red_ls += redundancy_ls.mean().asscalar()
                step_lab_ls += lab_loss.mean().asscalar() if lab_loss is not None else 0.0
                if (batch_id + 1) % (self.log_interval) == 0:
                    self._output_status(('[Epoch {}/{} Batch {}/{}] loss={:.4f}, recon_loss={:.4f}, ' \
                                         'kl_loss={:.4f}, red_loss={:.4f}, gen_lr={:.7f}, lab_loss={:.4f}')
                                 .format(epoch_id, self.epochs, batch_id + 1, len(dataloader),
                                         step_loss / self.log_interval, step_recon_ls / self.log_interval,
                                         step_kl_ls / self.log_interval, step_red_ls / self.log_interval,
                                         gen_trainer.learning_rate, step_lab_ls / self.log_interval))
                    step_loss = 0
                    step_recon_ls = 0
                    step_kl_ls = 0
                    step_lab_ls = 0
                if self.max_batches > 0 and step_num >= self.max_batches:
                    break
                
            mx.nd.waitall()
            if val_X is not None and (self.validate_each_epoch or epoch_id == self.epochs-1):
                v_res  = self.validate(model, bow_train, bow_val, val_y, dataloader_val)
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


class LabeledSeqBowEstimator(SeqBowEstimator):

    def __init__(self, n_labels, gamma, *args, igamma=1.0, multilabel=False, **kwargs):
        super(LabeledSeqBowEstimator, self).__init__(*args, **kwargs)
        self.multilabel = multilabel
        self.gamma = gamma
        self.n_labels = n_labels
        self.igamma = igamma


    @classmethod
    def from_config(cls, n_labels, gamma, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        est.n_labels = n_labels
        est.gamma    = gamma
        return est

    def _get_objective_from_validation_result(self, v_res):
        topic_obj = super()._get_objective_from_validation_result(v_res)
        acc = v_res['accuracy']
        return topic_obj + self.gamma * acc

    def _get_model(self):
        model = LabeledBertBowVED(self.n_labels, self.gamma, self.bert_base, self.vocabulary, self.latent_distrib,
                                  multilabel=self.multilabel,
                           n_latent=self.n_latent, 
                           kappa = self.kappa,
                           alpha = self.alpha,
                           batch_size = self.batch_size,
                                  kld=1.0, wd_freqs=self.wd_freqs, igamma = self.igamma,
                           redundancy_reg_penalty=self.redundancy_reg_penalty,
                           ctx=self.ctx)
        return model


    def validate(self, model, bow_train, bow_val_X, val_y, dataloader):
        v_res = super().validate(model, bow_train, bow_val_X, val_y, dataloader)
        tot_correct = 0
        tot = 0
        bs = min(bow_val_X.shape[0], self.batch_size)
        num_std_batches = bow_val_X.shape[0] // bs
        for i, seqs in enumerate(dataloader):
            if i < num_std_batches - 1:
                input_ids, valid_length, type_ids, output_vocab, labels = seqs
                predictions = self.model.predict(input_ids.as_in_context(self.ctx),
                                                 type_ids.as_in_context(self.ctx),
                                                 valid_length.astype('float32').as_in_context(self.ctx))
                labels = labels.as_in_context(self.ctx)
                correct = mx.nd.argmax(predictions, axis=1) == labels.squeeze()
                tot_correct += mx.nd.sum(correct).asscalar()
                tot += (input_ids.shape[0] - (labels < 0.0).sum().asscalar())
        acc = float(tot_correct) / float(tot)
        v_res['accuracy'] = acc
        return v_res


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

    
