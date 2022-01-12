# coding: utf-8
# Copyright (c) 2020-2021 The MITRE Corporation.
"""
Estimator module to train/fit/estimate individual models with fixed hyperparameters.
Estimators are used by trainers to manage training with specific datasets; in addition,
the estimator API supports inference/encoding with fitted models.
"""

import logging
import math
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
import umap
#import umap.plot
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, top_k_accuracy_score, roc_auc_score, ndcg_score, precision_recall_fscore_support
from tmnt.data_loading import DataIterLoader, SparseMatrixDataIter, PairedDataLoader, SingletonWrapperLoader
from tmnt.modeling import BowVAEModel, CovariateBowVAEModel, SeqBowVED
from tmnt.modeling import GeneralizedSDMLLoss, MetricSeqBowVED, MetricBowVAEModel
from tmnt.eval_npmi import EvaluateNPMI
from tmnt.distribution import HyperSphericalDistribution, LogisticGaussianDistribution, BaseDistribution, GaussianDistribution
import autogluon.core as ag
from itertools import cycle
import pickle
from typing import List, Tuple, Dict, Optional, Union, NoReturn

MAX_DESIGN_MATRIX = 250000000

def multilabel_pr_fn(cutoff, recall=False):

    def get_recall_or_precision(yvec, pvec):
        prec, rec, _, support = precision_recall_fscore_support(yvec, pvec, zero_division=0, average='samples')
        if recall:
            return rec
        else:
            return prec
    
    def multilabel_recall_fn_x(label, pred):
        num_labels = label[0].shape[0]
        pred_decision = np.where(pred >= cutoff, 1.0, 0.0)
        w_sum = get_recall_or_precision(label, pred_decision)
        return w_sum, label.shape[0]

    return multilabel_recall_fn_x

def get_composite_p_and_r_metric():
    metrics = mx.metric.CompositeEvalMetric()
    prec_metric = mx.metric.CustomMetric(feval=multilabel_pr_fn(0.5, recall=False))
    rec_metric = mx.metric.CustomMetric(feval=multilabel_pr_fn(0.5, recall=True))
    metrics.add(prec_metric)
    metrics.add(rec_metric)
    return metrics


class BaseEstimator(object):
    """Base class for all VAE-based estimators.
    
    Parameters:
        log_method: Method for logging. 'print' | 'log', optional (default='log')
        quiet: Flag for whether to force minimal logging/ouput. optional (default=False)
        coherence_coefficient: Weight to tradeoff influence of coherence vs perplexity in model 
            selection objective (default = 8.0)
        reporter: Callback reporter to include information for 
            model selection via AutoGluon
        ctx: MXNet context for the estimator
        latent_distribution: Latent distribution of the variational autoencoder - defaults to LogisticGaussian with 20 dimensions         
        optimizer: MXNet optimizer (default = "adam")
        lr: Learning rate of training. (default=0.005)
        coherence_reg_penalty: Regularization penalty for topic coherence. optional (default=0.0)
        redundancy_reg_penalty: Regularization penalty for topic redundancy. optional (default=0.0)
        batch_size: Batch training size. optional (default=128)
        epochs : Number of training epochs. optional(default=40)
        coherence_via_encoder: Flag to use encoder to derive coherence scores (via gradient attribution)
        pretrained_param_file: Path to pre-trained parameter file to initialize weights
        warm_start: Subsequent calls to `fit` will use existing model weights rather than reinitializing
    """
    def __init__(self,
                 log_method: str = 'log',
                 quiet: bool = False,
                 coherence_coefficient: float = 8.0,
                 reporter: Optional[object] = None,
                 ctx: Optional[mx.context.Context] = mx.cpu(),
                 latent_distribution: Optional[BaseDistribution] = None,
                 optimizer: str = "adam",
                 lr: float = 0.005, 
                 coherence_reg_penalty: float = 0.0,
                 redundancy_reg_penalty: float = 0.0,
                 batch_size: int = 128,
                 epochs: int = 40,
                 coherence_via_encoder: bool = False,
                 pretrained_param_file: Optional[str] = None,
                 warm_start: bool = False,
                 test_batch_size: int = 0):
        self.log_method = log_method
        self.quiet = quiet
        self.model = None
        self.coherence_coefficient = coherence_coefficient
        self.reporter = reporter
        self.ctx = ctx
        self.latent_distribution = latent_distribution or LogisticGaussianDistribution(20)
        self.optimizer = optimizer
        self.lr = lr
        self.n_latent = self.latent_distribution.n_latent
        self.coherence_reg_penalty = coherence_reg_penalty
        self.redundancy_reg_penalty = redundancy_reg_penalty
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.epochs = epochs
        self.coherence_via_encoder = coherence_via_encoder
        self.pretrained_param_file = pretrained_param_file
        self.warm_start = warm_start
        self.num_val_words = -1 ## will be set later for computing Perplexity on validation dataset
        self.latent_distribution.ctx = self.ctx


    def _np_one_hot(self, vec, n_outputs):
        ovec = np.zeros((vec.size, n_outputs))
        ovec[np.arange(vec.size), vec.astype('int32')] = 1.0
        return ovec
        

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


    def _npmi(self, X, k=10):
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
    

    def fit(self, X: sp.csr.csr_matrix, y: np.ndarray) -> NoReturn:
        """
        Fit VAE model according to the given training data X with optional co-variates y.
  
        Parameters:
            X: representing input data
            y: representing covariate/labels associated with data elements
        """
        raise NotImplementedError()
    

    def fit_with_validation(self, X: sp.csr.csr_matrix, y: np.ndarray, val_X: sp.csr.csr_matrix, val_Y: np.ndarray) -> NoReturn:
        """
        Fit VAE model according to the given training data X with optional co-variates y;
        validate (potentially each epoch) with validation data val_X and optional co-variates val_Y
  
        Parameters:
            X: representing training data
            y: representing covariate/labels associated with data elements in training data
            val_X: representing validation data
            val_y: representing covariate/labels associated with data elements in validation data
        """
        raise NotImplementedError()


class BaseBowEstimator(BaseEstimator):
    """
    Bag of words variational autoencoder algorithm

    Parameters:
        vocabulary (:class:`gluonnlp.Vocab`): GluonNLP Vocabulary object
        n_labels: Number of possible labels/classes when provided supervised data
        gamma: Coefficient that controls how supervised and unsupervised losses are weighted against each other
        enc_hidden_dim (int): Size of hidden encoder layers. optional (default=150)
        embedding_source (str): Word embedding source for vocabulary.
            'random' | 'glove' | 'fasttext' | 'word2vec', optional (default='random')
        embedding_size (int): Word embedding size, ignored if embedding_source not 'random'. optional (default=128)
        fixed_embedding (bool): Enable fixed embeddings. optional(default=False)
        num_enc_layers: Number of layers in encoder. optional(default=1)
        enc_dr: Dropout probability in encoder. optional(default=0.1)
        coherence_via_encoder: Flag 
        validate_each_epoch: Perform validation of model against heldout validation 
            data after each training epoch
        multilabel: Assume labels are vectors denoting label sets associated with each document
    """
    def __init__(self,
                 vocabulary: nlp.Vocab,
                 n_labels: int = 0,
                 gamma: float = 1.0,
                 multilabel: bool = False,
                 validate_each_epoch: bool = False,
                 enc_hidden_dim: int = 150,
                 embedding_source: str = "random",
                 embedding_size: int = 128,
                 fixed_embedding: bool = False,
                 num_enc_layers: int = 1,
                 enc_dr: float = 0.1,
                 classifier_dropout: float = 0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc_hidden_dim = enc_hidden_dim
        self.fixed_embedding = fixed_embedding
        self.n_encoding_layers = num_enc_layers
        self.enc_dr = enc_dr
        self.classifier_dropout = classifier_dropout
        self.vocabulary = vocabulary 
        self.embedding_source = embedding_source
        self.embedding_size = embedding_size
        self.validate_each_epoch = validate_each_epoch
        self.multilabel = multilabel
        self.gamma = gamma
        self.n_labels = n_labels
        self.has_classifier = n_labels > 1
        self.loss_function = gluon.loss.SigmoidBCELoss() if multilabel else gluon.loss.SoftmaxCELoss()

    @classmethod
    def from_saved(cls, model_dir: str, ctx: Optional[mx.context.Context] = mx.cpu()) -> 'BaseBowEstimator':
        """
        Instantiate a BaseBowEstimator object from a saved model

        Parameters:
            model_dir: String representing the path to the saved model directory
        Returns:
            BaseBowEstimator object
        """
        return cls.from_config(config     = model_dir+'/model.config',
                               vocabulary = model_dir+'/vocab.json',
                               pretrained_param_file = model_dir+'/model.params',
                               ctx        = ctx)

    @classmethod
    def from_config(cls, config: Union[str, dict], vocabulary: Union[str, nlp.Vocab],
                    n_labels: int = 0,
                    coherence_coefficient: float = 8.0,
                    coherence_via_encoder: bool = False,
                    validate_each_epoch: bool = False,
                    pretrained_param_file: Optional[str] = None,
                    reporter: Optional[object] = None,
                    ctx: mx.context.Context = mx.cpu()) -> 'BaseBowEstimator':
        """
        Create an estimator from a configuration file/object rather than by keyword arguments
        
        Parameters:
            config: Path to a json representation of a configuation or TMNT config dictionary
            vocabulary: Path to a json representation of a vocabulary or GluonNLP vocabulary object
            pretrained_param_file: Path to pretrained parameter file if using pretrained model
            reporter: Callback reporter to include information for model selection via AutoGluon
            ctx: MXNet context for the estimator

        Returns:
            An estimator for training and evaluation of a single model
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
        gamma = config.get('gamma', 1.0)
        multilabel = config.get('multilabel', False)
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
        if latent_distrib == 'logistic_gaussian':
            alpha = ldist_def.alpha
            latent_distribution = LogisticGaussianDistribution(n_latent, ctx=ctx, alpha=alpha)
        elif latent_distrib == 'vmf':
            kappa = ldist_def.kappa
            latent_distribution = HyperSphericalDistribution(n_latent, ctx=ctx, kappa=kappa)
        else:
            latent_distribution = GaussianDistribution(n_latent, ctx=ctx)
        n_labels = config.get('n_labels', n_labels)
        model = \
                cls(vocabulary,
                    n_labels=n_labels,
                    gamma = gamma,
                    multilabel = multilabel,
                    validate_each_epoch=validate_each_epoch,
                    coherence_coefficient=coherence_coefficient,
                    reporter=reporter, 
                    ctx=ctx, lr=lr, latent_distribution=latent_distribution, optimizer=optimizer,
                    enc_hidden_dim=enc_hidden_dim,
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
        config['n_labels']           = self.n_labels
        config['covar_net_layers']   = 1
        config['n_covars']           = 0
        if isinstance(self.latent_distribution, HyperSphericalDistribution):
            config['latent_distribution'] = {'dist_type':'vmf', 'kappa': self.latent_distribution.kappa}
        elif isinstance(self.latent_distribution, LogisticGaussianDistribution):
            config['latent_distribution'] = {'dist_type':'logistic_gaussian', 'alpha':self.latent_distribution.alpha}
        else:
            config['latent_distribution'] = {'dist_type':'gaussian'}
        if self.embedding_source != 'random':
            config['embedding'] = {'source': self.embedding_source}
        else:
            config['embedding'] = {'source': 'random', 'size': self.embedding_size}
        config['derived_info'] = {'embedding_size': self.embedding_size}
        return config
    

    def write_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
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
        for i, ((data,labels),) in enumerate(dataloader):
            data = data.as_in_context(self.ctx)
            _, kl_loss, rec_loss, _, _, _ = self._forward(self.model, data)
            if i == num_batches - 1 and last_batch_size > 0:
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
        test_batch_size = min(val_X.shape[0], self.test_batch_size)
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
        val_dataloader = SingletonWrapperLoader(val_dataloader)
        return val_dataloader

    def validate_with_loader(self, val_dataloader, val_size, total_val_words, val_X=None, val_y=None):
        ppl = self._perplexity(val_dataloader, total_val_words)
        if val_X is not None:
            n = min(val_X.shape[0], 50000)
            npmi, redundancy = self._npmi(val_X[:n])
        else:
            npmi, redundancy = self._npmi_with_dataloader(val_dataloader)
        v_res = {'ppl': ppl, 'npmi': npmi, 'redundancy': redundancy}
        prediction_arrays = []
        if self.has_classifier:
            tot_correct = 0
            tot = 0
            bs = min(val_size, self.batch_size)
            num_std_batches = val_size // bs
            last_batch_size = val_size % bs
            for i, ((data, labels),) in enumerate(val_dataloader):
                data = data.as_in_context(self.ctx)
                labels = labels.as_in_context(self.ctx)
                if i == num_std_batches - 1 and last_batch_size > 0:
                    data = data[:last_batch_size]
                    labels = labels[:last_batch_size]
                predictions = self.model.predict(data)    
                predictions_lists = [ p.asnumpy() for p in list(predictions) ]
                prediction_arrays.extend(predictions_lists)
                if len(labels.shape) == 1:  ## standard single-label classification
                    correct = mx.nd.argmax(predictions, axis=1) == labels
                    tot_correct += mx.nd.sum(correct).asscalar()
                tot += (data.shape[0] - (labels < 0.0).sum().asscalar()) # subtract off labels < 0 (for unlabeled data)
            acc = float(tot_correct) / float(tot)
            v_res['accuracy'] = acc
            prediction_mat = np.array(prediction_arrays)
            ap_scores = []
            if val_y is not None:
                if len(val_y.shape) == 1:
                    val_y = self._np_one_hot(val_y, self.n_labels)
                for c in range(self.n_labels):
                    y_vec = val_y[:,c]
                    pred_vec = prediction_mat[:,c]
                    if not np.any(np.isnan(pred_vec)):
                        ap_c = average_precision_score(y_vec, pred_vec)
                    else:
                        ap_c = 0.0
                    ap_scores.append((ap_c, int(y_vec.sum())))
            prediction_np_mat = np.array(prediction_arrays)
            v_res['ap_scores_and_support'] = ap_scores
        return v_res

    def validate(self, val_X, val_y):
        val_dataloader = self._get_val_dataloader(val_X, val_y)
        total_val_words = val_X.sum()
        if self.num_val_words < 0:
            self.num_val_words = total_val_words
        return self.validate_with_loader(val_dataloader, val_X.shape[0], total_val_words, val_X, val_y)


    def initialize_with_pretrained(self):
        raise NotImplementedError()


    def _get_objective_from_validation_result(self, val_result):
        npmi = val_result['npmi']
        ppl  = val_result['ppl']
        redundancy = val_result['redundancy']
        obj = (npmi - redundancy) - ( ( ppl / 1000 ) / self.coherence_coefficient )
        b_obj = max(min(obj, 100.0), -100.0)
        sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
        if self.has_classifier:
            orig_obj = sc_obj
            sc_obj = (sc_obj + self.gamma * val_result['accuracy']) / (1.0 + self.gamma)
            logging.info("Objective via classifier: {} based on accuracy = {} and topic objective = {}"
                         .format(sc_obj, val_result['accuracy'], orig_obj))
        else:
            logging.info("Pure topic model objective: {} (has classifier = {})".format(sc_obj, self.has_classifier))
        return sc_obj


    def _get_losses(self, model, batch_data):
        # batch_data has form: ((data, labels),)
        (data,labels), = batch_data
        data = data.as_in_context(self.ctx)
        if labels is None:
            labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
        labels = labels.as_in_context(self.ctx)
        
        elbo_ls, kl_ls, rec_ls, coherence_loss, red_ls, predicted_labels = \
            self._forward(self.model, data)
        if self.has_classifier:
            label_ls = self.loss_function(predicted_labels, labels).mean()
            total_ls = (self.gamma * label_ls) + elbo_ls.mean()
        else:
            total_ls = elbo_ls.mean()
            label_ls = mx.nd.zeros(total_ls.shape)
        return elbo_ls, kl_ls, rec_ls, red_ls, label_ls, total_ls

    def _get_unlabeled_losses(self, model, data):
        elbo_ls, kl_ls, rec_ls, coherence_loss, red_ls, predicted_labels = \
            self._forward(self.model, data)
        total_ls = elbo_ls.mean() / self.gamma
        return elbo_ls, kl_ls, rec_ls, red_ls, total_ls

    def fit_with_validation_loaders(self, train_dataloader, validation_dataloader, aux_dataloader,
                                    train_X_size, val_X_size, aux_X_size, total_val_words, val_X=None, val_y=None):
        all_model_params = self.model.collect_params()                
        params = [p for p in all_model_params.values() if p.grad_req != 'null']                
        for p in params:
            p.grad_req = 'add'
            
        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer, {'learning_rate': self.lr})
        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0
        v_res = None
        joint_loader = PairedDataLoader(train_dataloader, aux_dataloader)
        for epoch in range(self.epochs):
            ts_epoch = time.time()
            elbo_losses = []
            lab_losses  = []
            for i, (data_batch, aux_batch) in enumerate(joint_loader):
                with autograd.record():
                    elbo_ls, kl_loss, _, _, lab_loss, total_ls = self._get_losses(self.model, data_batch)
                    elbo_mean = elbo_ls.mean()
                total_ls.backward()

                if aux_batch is not None:
                    aux_data, = aux_batch
                    aux_data, _ = aux_data # ignore (null) label
                    aux_data = aux_data.as_in_context(self.ctx)
                    with autograd.record():
                        elbo_ls_a, kl_loss_a, _, _, total_ls_a = \
                            self._get_unlabeled_losses(self.model, aux_data)
                    total_ls_a.backward()
                
                trainer.allreduce_grads()
                trainer.update(1)
                all_model_params.zero_grad()
                if not self.quiet:
                    if aux_batch is not None:
                        elbo_losses.append(float(elbo_mean.asscalar()) + float(elbo_ls_a.mean().asscalar()))
                    else:
                        elbo_losses.append(float(elbo_mean.asscalar()))                        
                    if lab_loss is not None:
                        lab_losses.append(float(lab_loss.mean().asscalar()))
            if not self.quiet and not self.validate_each_epoch:
                elbo_mean = np.mean(elbo_losses) if len(elbo_losses) > 0 else 0.0
                lab_mean  = np.mean(lab_losses) if len(lab_losses) > 0 else 0.0
                self._output_status("Epoch [{}] finished in {} seconds. [elbo = {}, label loss = {}]"
                                    .format(epoch+1, (time.time()-ts_epoch), elbo_mean, lab_mean))
            mx.nd.waitall()
            if validation_dataloader is not None and (self.validate_each_epoch or epoch == self.epochs-1):
                sc_obj, v_res = self._perform_validation(epoch, validation_dataloader, val_X_size, total_val_words, val_X, val_y)
        mx.nd.waitall()
        if v_res is None:
            sc_obj, v_res = self._perform_validation(0, validation_dataloader, val_X_size, total_val_words, val_X, val_y)
        return sc_obj, v_res

    
    def _perform_validation(self,
                            epoch,
                            validation_dataloader,
                            val_X_size,
                            total_val_words,
                            val_X = None,
                            val_y = None):
        logging.info('Performing validation ....')
        v_res = self.validate_with_loader(validation_dataloader, val_X_size, total_val_words, val_X, val_y)
        sc_obj = self._get_objective_from_validation_result(v_res)
        if self.has_classifier:
            self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}. Accuracy = {}."
                                .format(epoch+1, sc_obj, v_res['ppl'],
                                        v_res['npmi'], v_res['redundancy'], v_res['accuracy']))
        else:
            self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                                .format(epoch+1, sc_obj, v_res['ppl'], v_res['npmi'], v_res['redundancy']))
        if self.reporter:
            self.reporter(epoch=epoch+1, objective=sc_obj, time_step=time.time(),
                          coherence=v_res['npmi'], perplexity=v_res['ppl'], redundancy=v_res['redundancy'])
        return sc_obj, v_res


    def setup_model_with_biases(self, X: sp.csr.csr_matrix) -> int:
        wd_freqs = self._get_wd_freqs(X)
        x_size = X.shape[0] * X.shape[1]
        if self.model is None or not self.warm_start:
            self.model = self._get_model()
            self.model.initialize_bias_terms(mx.nd.array(wd_freqs).squeeze())  ## initialize bias weights to log frequencies
        return x_size

    

    def fit_with_validation(self, 
                            X: sp.csr.csr_matrix,
                            y: np.ndarray,
                            val_X: Optional[sp.csr.csr_matrix],
                            val_y: Optional[np.ndarray],
                            aux_X: Optional[sp.csr.csr_matrix] = None) -> Tuple[float, dict]:
        """
        Fit a model according to the options of this estimator and optionally evaluate on validation data

        Parameters:
            X: Input training tensor
            y: Input labels/co-variates to use (optionally) for co-variate models
            val_X: Validateion input tensor
            val_y: Validation co-variates
            aux_X: Auxilliary unlabeled data for semi-supervised training

        Returns:
            sc_obj, v_res
        """
        
        x_size = self.setup_model_with_biases(X)
        
        if x_size > MAX_DESIGN_MATRIX:
            logging.info("Sparse matrix has total size = {}. Using Sparse Matrix data batcher.".format(x_size))
            train_dataloader = \
                DataIterLoader(SparseMatrixDataIter(X, y, batch_size = self.batch_size, last_batch_handle='discard', shuffle=True))
            train_X_size = X.shape[0]
        else:
            y = mx.nd.array(y) if y is not None else None
            X = mx.nd.sparse.csr_matrix(X)
            train_dataloader = DataIterLoader(mx.io.NDArrayIter(X, y, self.batch_size, last_batch_handle='discard', shuffle=True))
            train_X_size = X.shape[0]
        if aux_X is not None:
            aux_X_size = aux_X.shape[0] * aux_X.shape[1]
            if aux_X_size > MAX_DESIGN_MATRIX:
                aux_dataloader = \
                    DataIterLoader(SparseMatrixDataIter(aux_X, None, batch_size = self.batch_size, last_batch_handle='discard', shuffle=True))
            else:
                aux_X = mx.nd.sparse.csr_matrix(aux_X)
                aux_dataloader = \
                    DataIterLoader(mx.io.NDArrayIter(aux_X, None, self.batch_size, last_batch_handle='discard', shuffle=True))
        else:
            aux_dataloader, aux_X_size = None, 0
        if val_X is not None:
            val_dataloader = self._get_val_dataloader(val_X, val_y) # already wrapped in a singleton
            total_val_words = val_X.sum()
            val_X_size = val_X.shape[0]
        else:
            val_dataloader, total_val_words, val_X_size = None, 0, 0
        train_dataloader = SingletonWrapperLoader(train_dataloader)
        if aux_dataloader is not None:
            aux_dataloader   = SingletonWrapperLoader(aux_dataloader)        
        return self.fit_with_validation_loaders(train_dataloader, val_dataloader, aux_dataloader, train_X_size, val_X_size,
                                         aux_X_size, total_val_words, val_X=val_X, val_y=val_y)

                    
    def fit(self, X: sp.csr.csr_matrix, y: np.ndarray = None) -> 'BaseBowEstimator':
        """
        Fit VAE model according to the given training data X with optional co-variates y.
  
        Parameters:
            X: representing input data
            y: representing covariate/labels associated with data elements

        Returns:
            self
        """
        self.fit_with_validation(X, y, None, None)
        return self


class BowEstimator(BaseBowEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, *args, **kwargs):
        return super().from_config(*args, **kwargs)

    @classmethod
    def from_saved(cls, *args, **kwargs):
        return super().from_saved(*args, **kwargs)
    
    def npmi(self, X, k=10):
        return self._npmi(X, k=k)

    def perplexity(self, X: sp.csr.csr_matrix) -> float:
        """
        Calculate approximate perplexity for data X and y

        Parameters:
            X: Document word matrix of shape [n_samples, vocab_size]

        Returns:
           Perplexity score.
        """
        return super().perplexity(X, None)

    def _forward(self, model: BowVAEModel, data: mx.nd.NDArray):
        """
        Forward pass of BowVAE model given the supplied data

        Parameters:
            model: Core VAE model for bag-of-words topic model
            data: Document word matrix of shape (n_train_samples, vocab_size)

        Returns:
            Tuple of:
                elbo, kl_loss, rec_loss, coherence_loss, redundancy_loss, reconstruction
        """
        return model(data)


    def initialize_with_pretrained(self):
        assert(self.pretrained_param_file is not None)
        self.model = self._get_model()
        self.model.load_parameters(self.pretrained_param_file, allow_missing=False)


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
            for word in self.vocabulary.embedding._idx_to_token:
                if (self.vocabulary.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                    self.vocabulary.embedding[word] = mx.nd.random.normal(0, 0.1, emb_size)
        else:
            emb_size = self.embedding_size
        model = \
                BowVAEModel(self.enc_hidden_dim, emb_size, n_encoding_layers=self.n_encoding_layers,
                            enc_dr=self.enc_dr, fixed_embedding=self.fixed_embedding,
                            classifier_dropout=self.classifier_dropout,
                            n_labels = self.n_labels,
                            gamma = self.gamma,
                            multilabel = self.multilabel,
                            vocabulary=self.vocabulary, 
                            latent_distribution=self.latent_distribution, 
                            coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                            n_covars=0, ctx=self.ctx)
        if self.pretrained_param_file is not None:
            model.load_parameters(self.pretrained_param_file, allow_missing=False)
        return model
    

    def get_topic_vectors(self) -> mx.nd.NDArray:
        """
        Get topic vectors of the fitted model.

        Returns:
            topic_distribution: topic_distribution[i, j] represents word j in topic i. shape=(n_latent, vocab_size)
        """

        return self.model.get_topic_vectors() 

    def transform(self, X: sp.csr.csr_matrix) -> mx.nd.NDArray:
        """
        Transform data X according to the fitted model.

        Parameters:
            X: Document word matrix of shape {n_samples, n_features}

        Returns:
            topic_distribution: shape=(n_samples, n_latent) Document topic distribution for X
        """
        mx_array = mx.nd.array(X,dtype='float32')
        return self.model.encode_data(mx_array).asnumpy()


class BowMetricEstimator(BowEstimator):

    def __init__(self, *args, sdml_smoothing_factor=0.3, plot_dir=None, non_scoring_index=-1, **kwargs):
        super(BowMetricEstimator, self).__init__(*args, **kwargs)
        self.loss_function = GeneralizedSDMLLoss(smoothing_parameter=sdml_smoothing_factor)
        self.plot_dir = plot_dir
        self.non_scoring_index = non_scoring_index


    @classmethod
    def from_config(cls, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        return est

    def _get_model(self, bow_size=-1):
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
            MetricBowVAEModel(self.enc_hidden_dim, emb_size, n_encoding_layers=self.n_encoding_layers,
                            enc_dr=self.enc_dr, fixed_embedding=self.fixed_embedding,
                            classifier_dropout=self.classifier_dropout,
                            n_labels = self.n_labels,
                            gamma = self.gamma,
                            multilabel = self.multilabel,
                            vocabulary=self.vocabulary, 
                            latent_distribution=self.latent_distribution, 
                            coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                            n_covars=0, ctx=self.ctx)
        if self.pretrained_param_file is not None:
            model.load_parameters(self.pretrained_param_file, allow_missing=False)
        return model
        

    def _get_model_bias_initialize(self, train_data):
        model = self._get_model()
        tr_bow_matrix = self._get_bow_matrix(train_data)
        model.initialize_bias_terms(tr_bow_matrix.sum(axis=0))
        return model

    def _forward(self, model, data):
        elbo_ls, rec_ls, kl_ls, red_ls, total_ls = self._get_unlabeled_losses(model, data)
        return elbo_ls, rec_ls, kl_ls, red_ls, total_ls, None

    def _ff_batch(self, model, batch_data):
        (batch1, labels1), (batch2, labels2) = batch_data
        batch1 = batch1.as_in_context(self.ctx)
        batch2 = batch2.as_in_context(self.ctx)
        labels1 = labels1.as_in_context(self.ctx)
        labels2 = labels2.as_in_context(self.ctx)
        elbos_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2 = model(batch1, batch2)
        return elbos_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, labels1, labels2


    def _get_losses(self, model, batch_data):
        elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2 = self._ff_batch(model, batch_data)
        label_ls = self.loss_function(z_mu1, label1, z_mu2, label2)
        label_ls = label_ls.mean()
        total_ls = (self.gamma * label_ls) + elbo_ls.mean()
        return elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls
        

    def _get_unlabeled_losses(self, model, data):
        elbo_ls, rec_ls, kl_ls, red_ls = model.unpaired_input_forward(data)
        total_ls = elbo_ls / self.gamma
        return elbo_ls, rec_ls, kl_ls, red_ls, total_ls


    def classifier_validate(self, model, dataloader, epoch_id, include_predictions=True):
        posteriors = []
        ground_truth = []
        ground_truth_idx = []
        emb2 = None
        emb1 = []
        for batch_id, data_batch in enumerate(dataloader):
            elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2 = self._ff_batch(model, data_batch)
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
        labels = np.arange(posteriors[0].shape[0])        
        try:
            auroc = roc_auc_score(ground_truth, posteriors, average='weighted', labels=labels)
        except:
            auroc = 0.0
            logging.error('ROC computation failed')
        ap_scores = []
        wsum = 0.0
        tot  = 0.0
        for c in range(len(labels)):
            y_vec = ground_truth[:,c]
            pred_vec = posteriors[:,c]
            if not np.any(np.isnan(pred_vec)):
                ap_c = average_precision_score(y_vec, pred_vec)
            else:
                ap_c = 0.0
            if c != self.non_scoring_index:
                support = int(y_vec.sum())
                if support > 0:
                    wsum += ap_c * support
                    tot  += support
            ap_scores.append((ap_c, int(y_vec.sum())))
            
        avg_prec = wsum / tot
        ndcg = ndcg_score(ground_truth, posteriors)
        top_acc_1 = top_k_accuracy_score(ground_truth_idx, posteriors, k=1, labels=labels)        
        top_acc_2 = top_k_accuracy_score(ground_truth_idx, posteriors, k=2, labels=labels)
        top_acc_3 = top_k_accuracy_score(ground_truth_idx, posteriors, k=3, labels=labels)
        top_acc_4 = top_k_accuracy_score(ground_truth_idx, posteriors, k=4, labels=labels)
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
            
        if include_predictions:
            res_predictions = posteriors
            res_ground_truth = ground_truth
        else:
            res_predictions, res_ground_truth = None, None
        return {'avg_prec': avg_prec, 'top_1': top_acc_1, 'top_2': top_acc_2, 'top_3': top_acc_3, 'top_4': top_acc_4,
                'au_roc': auroc, 'ndcg': ndcg, 'ap_scores': ap_scores, 'results_predictions': res_predictions, 'results_ground_truth': res_ground_truth}    
        

    def _perform_validation(self,
                            epoch,
                            validation_dataloader,
                            val_X_size,
                            total_val_words,
                            val_X = None,
                            val_y = None):
        logging.info("Performing validation .. val_X_size = {}".format(val_X_size))
        v_res = self.classifier_validate(self.model, validation_dataloader, epoch)
        self._output_status("Epoch [{}]. Objective = {} ==> Avg. Precision = {}, AuROC = {}, NDCG = {} [acc@1= {}, acc@2={}, acc@3={}, acc@4={}]"
                            .format(epoch, v_res['avg_prec'], v_res['avg_prec'], v_res['au_roc'], v_res['ndcg'],
                                    v_res['top_1'], v_res['top_2'], v_res['top_3'], v_res['top_4']))
        self._output_status("  AP Scores: {}".format(v_res['ap_scores']))
        if self.reporter:
            self.reporter(epoch=epoch+1, objective=v_res['avg_prec'], time_step=time.time(), coherence=0.0,
                          perplexity=0.0, redundancy=0.0)
        return v_res['avg_prec'], v_res



class CovariateBowEstimator(BaseBowEstimator):

    def __init__(self, *args, n_covars=0, **kwargs):

        super().__init__(*args, **kwargs)

        self.covar_net_layers = 1 ### XXX - temp hardcoded
        self.n_covars = n_covars


    @classmethod
    def from_config(cls, n_covars, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        est.n_covars = n_covars
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
                                 vocabulary=self.vocabulary, enc_dim=self.enc_hidden_dim, embedding_size=emb_size,
                                 fixed_embedding=self.fixed_embedding, latent_distribution=self.latent_distribution,
                                 coherence_reg_penalty=self.coherence_reg_penalty, redundancy_reg_penalty=self.redundancy_reg_penalty,
                                 batch_size=self.batch_size, n_encoding_layers=self.n_encoding_layers, enc_dr=self.enc_dr,
                                 ctx=self.ctx)
        return model


    def _get_losses(self, model, batch_data):
        # batch_data has form: ((data, covars),)
        (data,covars), = batch_data
        data = data.as_in_context(self.ctx)
        covars = covars.as_in_context(self.ctx)        
        elbo_ls, kl_ls, rec_ls, coherence_loss, red_ls, predicted_labels = \
            self._forward(self.model, data, covars)
        total_ls = elbo_ls.mean()
        label_ls = mx.nd.zeros(total_ls.shape)
        return elbo_ls, kl_ls, rec_ls, red_ls, label_ls, total_ls
    

    def _get_config(self):
        config = super()._get_config()
        config['n_covars'] = self.n_covars
        return config
    
    
    def _forward(self,
                 model: BowVAEModel,
                 data: mx.nd.NDArray,
                 covars: mx.nd.NDArray) -> Tuple[mx.nd.NDArray,
                                                 mx.nd.NDArray,
                                                 mx.nd.NDArray,
                                                 mx.nd.NDArray,
                                                 mx.nd.NDArray,
                                                 mx.nd.NDArray,
                                                 mx.nd.NDArray] :
        """
        Forward pass of BowVAE model given the supplied data

        Parameters:
            model: Model that returns elbo, kl_loss, rec_loss, l1_pen, coherence_loss, redundancy_loss, reconstruction
            data: Document word matrix of shape (n_train_samples, vocab_size) 
            covars: Covariate matrix. shape [n_samples, n_covars]

        Returns:
            (tuple): Tuple of: 
                elbo, kl_loss, rec_loss, l1_pen, coherence_loss, redundancy_loss, reconstruction
        """
        self.train_data = data
        self.train_labels = covars
        return model(data, covars)


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

    def _npmi(self, X, k=10):
        return super()._npmi(X, k=k)
        #return self._npmi_per_covariate(X, y, k)

    def _get_objective_from_validation_result(self, v_res):
        return v_res['npmi']

    def validate(self, X, y):
        npmi, redundancy = self._npmi(X)
        return {'npmi': npmi, 'redundancy': redundancy, 'ppl': 0.0}

    def get_topic_vectors(self) -> mx.nd.NDArray:
        """
        Get topic vectors of the fitted model.

        Returns:
            topic_vectors: Topic word distribution. topic_distribution[i, j] represents word j in topic i. 
                shape=(n_latent, vocab_size)
        """

        return self.model.get_topic_vectors(self.train_data, self.train_labels)

    def initialize_with_pretrained(self):
        assert(self.pretrained_param_file is not None)
        self.model = self._get_model()
        self.model.load_parameters(self.pretrained_param_file, allow_missing=False)
        

    def transform(self, X: sp.csr.csr_matrix, y: np.ndarray):
        """
        Transform data X and y according to the fitted model.

        Parameters:
            X: Document word matrix of shape {n_samples, n_features)
            y: Covariate matrix of shape (n_train_samples, n_covars)

        Returns:
            Document topic distribution for X and y of shape=(n_samples, n_latent)
        """
        x_mxnet, y_mxnet = mx.nd.array(X, dtype=np.float32), mx.nd.array(y, dtype=np.float32)
        return self.model.encode_data_with_covariates(x_mxnet, y_mxnet).asnumpy()
    

class SeqBowEstimator(BaseEstimator):

    def __init__(self, bert_base, bert_vocab, *args,
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
                 pure_classifier_objective = False,
                 validate_each_epoch = False,
                 **kwargs):
        super(SeqBowEstimator, self).__init__(*args, optimizer=optimizer, **kwargs)
        self.pure_classifier_objective = pure_classifier_objective
        self.validate_each_epoch = validate_each_epoch
        self.minimum_lr = 1e-9
        self.checkpoint_dir = checkpoint_dir
        self.bert_base = bert_base
        self.bert_vocab = bert_vocab
        self.bert_model_name = bert_model_name
        self.bert_data_name = bert_data_name
        self.has_classifier = n_labels >= 2
        self.classifier_dropout = classifier_dropout
        self.multilabel = multilabel
        self.n_labels = n_labels
        self.metric = get_composite_p_and_r_metric() if multilabel else mx.metric.Accuracy()
        self.warmup_ratio = warmup_ratio
        self.log_interval = log_interval
        self.loss_function = gluon.loss.SigmoidBCELoss() if multilabel else gluon.loss.SoftmaxCELoss(sparse_label=False)
        self.gamma = gamma
        self.decoder_lr = decoder_lr
        self._bow_matrix = None
        self.bow_vocab = bow_vocab


    @classmethod
    def from_config(cls,
                    config: Union[str, dict, ag.space.Dict],
                    bert_base: nlp.model.bert.BERTModel,
                    bert_vocab: nlp.Vocab,
                    bow_vocab: nlp.Vocab,
                    reporter: Optional[object] = None,
                    log_interval: int = 1,
                    pretrained_param_file: Optional[str] = None,
                    n_labels: Optional[int] = None,                    
                    ctx: mx.context.Context = mx.cpu()) -> 'SeqBowEstimator':
        """
        Instantiate an object of this class using the provided `config`

        Parameters:
            config: String to configuration path (in json format) or an autogluon dictionary representing the config
            bert_base: GluonNLP BERT model
            bow_vocab: Bag-of-words vocabulary used for decoding reconstruction target
            repoter: Autogluon reporter object with callbacks for logging model selection
            log_interval: Logging frequency (default = 1)
            pretrained_param_file: Parameter file
            ctx: MXNet context
        
        Returns:
            An object of this class
        """
        if isinstance(config, str):
            try:
                with open(config, 'r') as f:
                    config = json.load(f)
            except:
                logging.error("File {} does not appear to be a valid config instance".format(config))
                raise Exception("Invalid Json Configuration File")
        if isinstance(config, dict):
            config = ag.space.Dict(**config)
        ldist_def = config['latent_distribution']
        kappa = 0.0
        alpha = 1.0
        latent_distrib = ldist_def.dist_type
        n_latent = int(config.n_latent)
        if latent_distrib == 'logistic_gaussian':
            alpha = ldist_def.alpha
            latent_distribution = LogisticGaussianDistribution(n_latent, ctx=ctx, alpha=alpha)
        elif latent_distrib == 'vmf':
            kappa = ldist_def.kappa
            latent_distribution = HyperSphericalDistribution(n_latent, ctx=ctx, kappa=kappa)
        else:
            latent_distribution = GaussianDistribution(n_latent, ctx=ctx)
        estimator = cls(bert_base, bert_vocab, 
                        bert_model_name = config.bert_model_name,
                        bert_data_name  = config.bert_data_name,
                        bow_vocab       = bow_vocab, 
                        n_labels        = config.get('n_labels', n_labels),
                        latent_distribution = latent_distribution,
                        batch_size      = int(config.batch_size),
                        redundancy_reg_penalty = 0.0,
                        warmup_ratio = config.warmup_ratio,
                        optimizer = config.optimizer,
                        classifier_dropout = config.classifier_dropout,
                        epochs = int(config.epochs),
                        gamma = config.gamma,
                        lr = config.lr,
                        decoder_lr = config.decoder_lr,
                        pretrained_param_file = pretrained_param_file,
                        warm_start = (pretrained_param_file is not None),
                        reporter=reporter,
                        ctx=ctx,
                        log_interval=log_interval)
        estimator.initialize_with_pretrained()
        return estimator

    @classmethod
    def from_saved(cls, model_dir: str,
                   reporter: Optional[object] = None,
                   log_interval: int = 1,
                   ctx: Optional[mx.context.Context] = mx.cpu()) -> 'SeqBowEstimator':
        if model_dir is not None:
            param_file = os.path.join(model_dir, 'model.params')
            vocab_file = os.path.join(model_dir, 'vocab.json')
            config_file = os.path.join(model_dir, 'model.config')
            serialized_vectorizer_file = os.path.join(model_dir, 'vectorizer.pkl')
        with open(config_file) as f:
            config = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        bow_vocab = nlp.Vocab.from_json(voc_js)
        bert_base, bert_vocab = nlp.model.get_model(config['bert_model_name'],  
                                               dataset_name=config['bert_data_name'],
                                               pretrained=True, ctx=ctx, use_pooler=True,
                                               use_decoder=False, use_classifier=False)
        return cls.from_config(config,
                               bert_base = bert_base,
                               bert_vocab = bert_vocab,
                               bow_vocab = bow_vocab,
                               reporter = reporter,
                               log_interval = log_interval,
                               pretrained_param_file = param_file,
                               ctx = ctx)
    

    def initialize_with_pretrained(self):
        self.model = self._get_model()
        if self.pretrained_param_file is not None:
            self.model.load_parameters(self.pretrained_param_file, allow_missing=False)


    def _get_model_bias_initialize(self, train_data):
        model = self._get_model()
        tr_bow_counts = self._get_bow_wd_counts(train_data)
        model.initialize_bias_terms(tr_bow_counts)
        return model
        
    
    def _get_model(self):
        model = SeqBowVED(self.bert_base, self.latent_distribution, num_classes=self.n_labels, n_latent=self.n_latent,
                          bow_vocab_size = len(self.bow_vocab), dropout=self.classifier_dropout, ctx=self.ctx)
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
        if isinstance(self.latent_distribution, HyperSphericalDistribution):
            config['latent_distribution'] = {'dist_type':'vmf', 'kappa': self.latent_distribution.kappa}
        elif isinstance(self.latent_distribution, LogisticGaussianDistribution):
            config['latent_distribution'] = {'dist_type':'logistic_gaussian', 'alpha':self.latent_distribution.alpha}
        else:
            config['latent_distribution'] = {'dist_type':'gaussian'}
        config['epochs'] = self.epochs
        #config['embedding_source'] = self.embedding_source
        config['gamma'] = self.gamma
        config['redundancy_reg_penalty'] = self.redundancy_reg_penalty
        config['warmup_ratio'] = self.warmup_ratio
        config['bert_model_name'] = self.bert_model_name
        config['bert_data_name'] = self.bert_data_name
        config['classifier_dropout'] = self.classifier_dropout
        return config

    def write_model(self, model_dir: str, suffix: str ='') -> None:
        """
        Writes the model within this estimator to disk.

        Parameters:
            model_dir: Output directory for model parameters, config and vocabulary
            suffix: Suffix to use for model (e.g. at different checkpoints)
        """
        pfile = os.path.join(model_dir, ('model.params' + suffix))
        conf_file = os.path.join(model_dir, ('model.config' + suffix))
        vocab_file = os.path.join(model_dir, ('vocab.json' + suffix))
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
        if self.has_classifier:
            metric_nm, metric_val = metric.get()
            if not isinstance(metric_nm, list):
                metric_nm, metric_val = [metric_nm], [metric_val]
            self._output_status("Epoch {} Batch {}/{} loss={}, (rec_loss = {}), (red_loss = {}), (class_loss = {}) lr={:.10f}, metrics[{}]: {}"
                                .format(epoch_id+1, batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval, red_loss/log_interval,
                                        class_loss/log_interval, learning_rate, metric_nm, metric_val))
        else:
            self._output_status("Epoch {} Batch {}/{} loss={}, (rec_loss = {}), (red_loss = {}), (class_loss = {}) lr={:.10f}"
                                .format(epoch_id+1, batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval, red_loss/log_interval,
                                        class_loss/log_interval, learning_rate))

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
        for i, data in enumerate(dataloader):
            seqs, = data
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
        for i, data in enumerate(dataloader):
            seqs, = data
            bow_batch = seqs[3].squeeze(axis=1)
            sums += bow_batch.sum(axis=0)
        return sums

    def _get_objective_from_validation_result(self, val_result):
        npmi = val_result['npmi']
        ppl  = val_result['ppl']
        redundancy = val_result['redundancy']
        obj = (npmi - redundancy) - ( ( ppl / 1000 ) / self.coherence_coefficient )
        b_obj = max(min(obj, 100.0), -100.0)
        sc_obj = 1.0 / (1.0 + math.exp(-b_obj))
        if self.has_classifier and self.gamma >= 0.0:
            orig_obj = sc_obj
            sc_obj = val_result['accuracy'] if self.pure_classifier_objective else (sc_obj + self.gamma * val_result['accuracy']) / (1.0 + self.gamma)
            logging.info("Objective via classifier: {} based on accuracy = {} and topic objective = {}"
                         .format(sc_obj, val_result['accuracy'], orig_obj))
        return sc_obj

    def _get_losses(self, model, batch_data):
        ## batch_data should be a singleton tuple: (seqs,)
        seqs, = batch_data
        input_ids, valid_length, type_ids, bow, label = seqs
        elbo_ls, rec_ls, kl_ls, red_ls, out = model(
            input_ids.as_in_context(self.ctx), type_ids.as_in_context(self.ctx),
            valid_length.astype('float32').as_in_context(self.ctx), bow.as_in_context(self.ctx))
        if self.has_classifier:
            label = label.as_in_context(self.ctx)
            label_ls = self.loss_function(out, label)
            label_ls = label_ls.mean()
            total_ls = (self.gamma * label_ls) + elbo_ls.mean()
            ## update label metric (e.g. accuracy)
            if not self.multilabel:
                label_ind = label.argmax(axis=1)
                self.metric.update(labels=[label_ind], preds=[out])
            else:
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
        

    def fit_with_validation(self,
                            train_data: gluon.data.DataLoader,
                            dev_data: gluon.data.DataLoader,
                            aux_data: gluon.data.DataLoader,
                            num_train_examples: int):
        """
        Training function.

        Parameters:
            train_data: Gluon dataloader with training data.
            dev_data: Gluon dataloader with dev/validation data.
            aux_data: Gluon dataloader with auxilliary data.
            num_train_examples: Number of training samples
        """
        if self.model is None or not self.warm_start:
            self.model = self._get_model_bias_initialize(train_data)

        model = self.model

        has_aux_data = aux_data is not None
        
        accumulate = False
        v_res      = None

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

        num_effective_samples = num_train_examples

        #step_size = self.batch_size * accumulate if accumulate else self.batch_size
        #num_train_steps = int((num_effective_samples / step_size) * self.epochs) + 1

        joint_loader = PairedDataLoader(train_data, aux_data)
        
        num_train_steps = len(joint_loader) * self.epochs
        if accumulate:
            num_train_steps /= accumulate
        
        warmup_ratio = self.warmup_ratio
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        logging.info("Number of warmup steps = {}, num total train steps = {}, train examples = {}, batch_size = {}, epochs = {}"
                     .format(num_warmup_steps, num_train_steps, num_effective_samples, self.batch_size, self.epochs))
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
        if (accumulate and accumulate > 1) or has_aux_data:
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
            
            for (batch_id, (data, aux_batch)) in enumerate(joint_loader):
                # data_batch is either a 2-tuple of: (labeled, unlabeled)
                # OR a 1-tuple of (labeled,)
                
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
                    elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls = self._get_losses(model, data)
                total_ls.backward()
                if aux_batch is not None:
                    with mx.autograd.record():
                        elbo_ls_2, rec_ls_2, kl_ls_2, red_ls_2, total_ls_2 = self._get_unlabeled_losses(model, aux_batch)
                    total_ls_2.backward()
                update_loss_details(total_ls, elbo_ls, red_ls, label_ls)
                if aux_batch is not None:
                    update_loss_details(total_ls_2, elbo_ls_2, red_ls_2, None)
                # update
                if not accumulate or (batch_id + 1) % accumulate == 0:
                    trainer.allreduce_grads()
                    dec_trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(clipped_params, 1.0, check_isfinite=True)
                    trainer.update(accumulate if accumulate else 1)
                    dec_trainer.update(accumulate if accumulate else 1)
                    step_num += 1
                    if (accumulate and accumulate > 1) or aux_batch:
                        # set grad to zero for gradient accumulation
                        all_model_params.zero_grad()
                if (batch_id + 1) % (self.log_interval) == 0:
                    self.log_train(batch_id, num_train_steps / self.epochs, self.metric, loss_details['step_loss'],
                                   loss_details['elbo_loss'], loss_details['red_loss'], loss_details['class_loss'], self.log_interval,
                                   epoch_id, trainer.learning_rate)
                    ## reset loss details
                    for d in loss_details:
                        loss_details[d] = 0.0
            mx.nd.waitall()

            # inference on dev data
            if dev_data is not None and (self.validate_each_epoch or epoch_id == (self.epochs-1)):
                sc_obj, v_res = self._perform_validation(model, dev_data, epoch_id)
            else:
                sc_obj, v_res = None, None
            if self.checkpoint_dir:
                self.write_model(self.checkpoint_dir, suffix=str(epoch_id))
        mx.nd.waitall()
        if v_res is None:
            sc_obj, v_res = self._perform_validation(model, dev_data, 0)
        return sc_obj, v_res


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
                logging.debug('All loss terms: {}, {}, {}, {}, {}, {}'.format(elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls))
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

    def __init__(self, *args, sdml_smoothing_factor=0.3, plot_dir=None, non_scoring_index=-1, **kwargs):
        super(SeqBowMetricEstimator, self).__init__(*args, **kwargs)
        self.loss_function = GeneralizedSDMLLoss(smoothing_parameter=sdml_smoothing_factor, x2_downweight_idx=non_scoring_index)
        self.plot_dir = plot_dir
        self.non_scoring_index = non_scoring_index ## if >=0 this will avoid considering this label index in evaluation


    @classmethod
    def from_config(cls, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        return est
        
    def _get_model(self, bow_size=-1):
        bow_size = bow_size if bow_size > 1 else len(self.bow_vocab)
        model = MetricSeqBowVED(self.bert_base, self.latent_distribution, n_latent=self.n_latent,
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
            batch_1, batch_2 = seqs                
            bow_matrix.extend(list(batch_2[3].squeeze(axis=1)))
            bow_matrix.extend(list(batch_1[3].squeeze(axis=1)))
        bow_matrix = mx.nd.stack(*bow_matrix)
        if cache:
            self._bow_matrix = bow_matrix
        return bow_matrix

    def _ff_batch(self, model, batch_data):
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
        ## convert back to label indices rather than 1-hot vecs
        label1_ind = label1.argmax(axis=1)
        label2_ind = label2.argmax(axis=1)
        label1 = label1_ind.as_in_context(self.ctx)
        label2 = label2_ind.as_in_context(self.ctx)
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
    
    def classifier_validate(self, model, dataloader, epoch_id, include_predictions=True):
        posteriors = []
        ground_truth = []
        ground_truth_idx = []
        emb2 = None
        emb1 = []
        for batch_id, data_batch in enumerate(dataloader):
            elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2 = self._ff_batch(model, data_batch)
            label1_ind = label1.argmax(axis=1)
            label2_ind = label2.argmax(axis=1)
            label1_ind = label1_ind.as_in_context(self.ctx)
            label2_ind = label2_ind.as_in_context(self.ctx)
            label_mat = self.loss_function._compute_labels(mx.ndarray, label1_ind, label2_ind)        
            dists = self.loss_function._compute_distances(z_mu1, z_mu2)
            probs = mx.nd.softmax(-dists, axis=1).asnumpy()
            posteriors += list(probs)
            ground_truth_idx += list(label1_ind.asnumpy()) ## index values for labels
            ground_truth += list(label1.asnumpy())
            if emb2 is None:
                emb2 = z_mu2.asnumpy()
            emb1 += list(z_mu1.asnumpy())
        posteriors = np.array(posteriors)        
        ground_truth = np.array(ground_truth)
        ground_truth_idx = np.array(ground_truth_idx)
        labels = np.arange(posteriors[0].shape[0])
        if not np.any(np.isnan(posteriors)):
            avg_prec = average_precision_score(ground_truth, posteriors, average='weighted')
        else:
            avg_prec = 0.0
        try:
            auroc = roc_auc_score(ground_truth, posteriors, average='weighted', labels=labels)
        except:
            auroc = 0.0
            logging.error('ROC computation failed')
        ndcg = ndcg_score(ground_truth, posteriors)

        ## bit of hackery to only compute average precision for non-null/other categories
        ## in many cases, we'd like to optimize over this score
        if self.non_scoring_index >= 0:
            wsum = 0.0
            tot  = 0.0
            for c in range(len(labels)):
                y_vec = ground_truth[:,c]
                pred_vec = posteriors[:,c]
                ap_c = average_precision_score(y_vec, pred_vec) if not np.any(np.isnan(pred_vec)) else 0.0
                if c != self.non_scoring_index:
                    support = int(y_vec.sum())
                    if support > 0:
                        wsum += ap_c * support
                        tot  += support
            avg_prec = wsum / tot

        top_acc_1 = top_k_accuracy_score(ground_truth_idx, posteriors, k=1, labels=labels)        
        top_acc_2 = top_k_accuracy_score(ground_truth_idx, posteriors, k=2, labels=labels)
        top_acc_3 = top_k_accuracy_score(ground_truth_idx, posteriors, k=3, labels=labels)
        top_acc_4 = top_k_accuracy_score(ground_truth_idx, posteriors, k=4, labels=labels)
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
        if include_predictions:
            res_predictions = posteriors
            res_ground_truth = ground_truth
        else:
            res_predictions, res_ground_truth = None, None
        return {'avg_prec': avg_prec, 'top_1': top_acc_1, 'top_2': top_acc_2, 'top_3': top_acc_3, 'top_4': top_acc_4,
                'au_roc': auroc, 'ndcg': ndcg, 'results_predictions': res_predictions, 'results_ground_truth': res_ground_truth}

            
    def _perform_validation(self, model, dev_data, epoch_id):
        v_res = self.classifier_validate(model, dev_data, epoch_id)
        self._output_status("Epoch [{}]. Objective = {} ==> Avg. Precision = {}, AuROC = {}, NDCG = {} [acc@1= {}, acc@2={}, acc@3={}, acc@4={}]"
                            .format(epoch_id, v_res['avg_prec'], v_res['avg_prec'], v_res['au_roc'], v_res['ndcg'],
                                    v_res['top_1'], v_res['top_2'], v_res['top_3'], v_res['top_4']))
        if self.reporter:
            self.reporter(epoch=epoch_id+1, objective=v_res['avg_prec'], time_step=time.time(), coherence=0.0,
                          perplexity=0.0, redundancy=0.0)
        return v_res['avg_prec'], v_res

