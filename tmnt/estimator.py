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
import numpy as np
import scipy.sparse as sp
import json

from sklearn.metrics import average_precision_score, top_k_accuracy_score, roc_auc_score, ndcg_score, precision_recall_fscore_support
from tmnt.data_loading import PairedDataLoader, SingletonWrapperLoader, SparseDataLoader, get_llm_model
from tmnt.modeling import BowVAEModel, SeqBowVED, BaseVAE
from tmnt.modeling import CrossBatchCosineSimilarityLoss, GeneralizedSDMLLoss, MultiNegativeCrossEntropyLoss, MetricSeqBowVED, MetricBowVAEModel
from tmnt.eval_npmi import EvaluateNPMI
from tmnt.distribution import LogisticGaussianDistribution, BaseDistribution, GaussianDistribution, VonMisesDistribution

## evaluation routines
from torcheval.metrics import MultilabelAUPRC, MulticlassAUPRC

## huggingface specifics
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.optimization import AdamW, get_scheduler

## model selection
import optuna

from itertools import cycle
import pickle
from typing import List, Tuple, Dict, Optional, Union, NoReturn

import torch
import torchtext
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

MAX_DESIGN_MATRIX = 250000000


class BaseEstimator(object):
    """Base class for all VAE-based estimators.
    
    Parameters:
        log_method: Method for logging. 'print' | 'log', optional (default='log')
        quiet: Flag for whether to force minimal logging/ouput. optional (default=False)
        coherence_coefficient: Weight to tradeoff influence of coherence vs perplexity in model 
            selection objective (default = 8.0)
        device: pytorch device
        latent_distribution: Latent distribution of the variational autoencoder - defaults to LogisticGaussian with 20 dimensions
        optimizer: optimizer (default = "adam")
        lr: Learning rate of training. (default=0.005)
        batch_size: Batch training size. optional (default=128)
        epochs : Number of training epochs. optional(default=40)
        coherence_via_encoder: Flag to use encoder to derive coherence scores (via gradient attribution)
        pretrained_param_file: Path to pre-trained parameter file to initialize weights
        warm_start: Subsequent calls to `fit` will use existing model weights rather than reinitializing
    """
    def __init__(self,
                 vocabulary = None,
                 log_method: str = 'log',
                 quiet: bool = False,
                 coherence_coefficient: float = 8.0,
                 device: Optional[str] = 'cpu',
                 latent_distribution: BaseDistribution = None,
                 lr: float = 0.005, 
                 batch_size: int = 128,
                 epochs: int = 40,
                 coherence_via_encoder: bool = False,
                 pretrained_param_file: Optional[str] = None,
                 warm_start: bool = False,
                 npmi_matrix: Optional[torch.Tensor] = None,
                 npmi_lambda: float = 0.7,
                 npmi_scale: float = 100.0,
                 test_batch_size: int = 0):
        self.vocabulary = vocabulary
        self.log_method = log_method
        self.quiet = quiet
        self.model : Optional[BaseVAE] = None
        self.coherence_coefficient = coherence_coefficient
        self.device = device
        self.latent_distribution = latent_distribution
        self.lr = lr
        self.n_latent = self.latent_distribution.n_latent
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.epochs = epochs
        self.coherence_via_encoder = coherence_via_encoder
        self.pretrained_param_file = pretrained_param_file
        self.warm_start = warm_start
        self.num_val_words = -1 ## will be set later for computing Perplexity on validation dataset
        self.latent_distribution.device = self.device
        self.npmi_matrix : Optional[torch.Tensor] = npmi_matrix ## used with NPMI loss
        self.npmi_lambda = npmi_lambda
        self.npmi_scale  = npmi_scale


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
        unique_limit = k  ## limit is the same as 'k'
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
    

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> NoReturn:
        """
        Fit VAE model according to the given training data X with optional co-variates y.
  
        Parameters:
            X: representing input data
            y: representing covariate/labels associated with data elements
        """
        raise NotImplementedError()
    

    def fit_with_validation(self, X: torch.Tensor, y: torch.Tensor, val_X: torch.Tensor, val_Y: torch.Tensor) -> NoReturn:
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
        self.embedding_source = embedding_source
        self.embedding_size = embedding_size
        self.validate_each_epoch = validate_each_epoch
        self.multilabel = multilabel
        self.gamma = gamma
        self.n_labels = n_labels
        self.has_classifier = n_labels > 1
        self.loss_function = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()

    @classmethod
    def from_saved(cls, model_dir: str, device: Optional[str] = 'cpu') -> 'BaseBowEstimator':
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
                               device        = device)

    @classmethod
    def from_config(cls, config: Union[str, dict], vocabulary: Union[str, torchtext.vocab.Vocab],
                    n_labels: int = 0,
                    coherence_coefficient: float = 8.0,
                    coherence_via_encoder: bool = False,
                    validate_each_epoch: bool = False,
                    pretrained_param_file: Optional[str] = None,
                    device: str = 'cpu') -> 'BaseBowEstimator':
        """
        Create an estimator from a configuration file/object rather than by keyword arguments
        
        Parameters:
            config: Path to a json representation of a configuation or TMNT config dictionary
            vocabulary: Path to a json representation of a vocabulary or vocabulary object
            pretrained_param_file: Path to pretrained parameter file if using pretrained model
            device: PyTorch Device

        Returns:
            An estimator for training and evaluation of a single model
        """
        if isinstance(config, str):
            try:
                with open(config, 'r') as f:
                    config = json.load(f)
            except:
                logging.error("File {} does not appear to be a valid config instance".format(config))
                raise Exception("Invalid Json Configuration File")
            #config = ag.space.Dict(**config_dict)
        if isinstance(vocabulary, str):
            try:
                with open(vocabulary, 'r') as f:
                    _voc = json.load(f)
                    voc_js = {k: 1 for k in _voc.keys()}
            except:
                logging.error("File {} does not appear to be a valid vocabulary file".format(vocabulary))
                raise Exception("Invalid Json Configuration File")            
            vocabulary = torchtext.vocab.vocab(voc_js)
        #if vocabulary['embedding'] is not None:
        if False:
            raise Exception("Pre-trained embeddings not yet (re-)supported")
            #emb_size = vocabulary['embedding'].idx_to_vec[0].size
        else:
            emb_size = config['embedding'].get('size')
            if not emb_size:
                emb_size = config['derived_info'].get('embedding_size')
            if not emb_size:
                raise Exception("Embedding size must be provided as the 'size' attribute of 'embedding' or as 'derived_info.embedding_size'")
        gamma = config.get('gamma', 1.0)
        multilabel = config.get('multilabel', False)
        lr = config['lr']
        latent_distrib = config['latent_distribution']
        n_latent = int(config['n_latent'])
        enc_hidden_dim = int(config['enc_hidden_dim'])
        batch_size = int(config['batch_size'])
        embedding_source = config['embedding']['source']
        fixed_embedding  = config['embedding'].get('fixed') == True
        n_encoding_layers = config['num_enc_layers']
        enc_dr = config['enc_dr']
        epochs = int(config['epochs'])
        ldist_def = config['latent_distribution']
        kappa = 0.0
        alpha = 1.0
        latent_distrib = ldist_def['dist_type']
        if latent_distrib == 'logistic_gaussian':
            alpha = ldist_def['alpha']
            latent_distribution = LogisticGaussianDistribution(enc_hidden_dim, n_latent, device=device, alpha=alpha)
        elif latent_distrib == 'vmf':
            kappa = ldist_def['kappa']
            latent_distribution = VonMisesDistribution(enc_hidden_dim, n_latent, device=device, kappa=kappa)
        else:
            latent_distribution = GaussianDistribution(enc_hidden_dim, n_latent, device=device)
        n_labels = config.get('n_labels', n_labels)
        model = \
                cls(vocabulary=vocabulary,
                    n_labels=n_labels,
                    gamma = gamma,
                    multilabel = multilabel,
                    validate_each_epoch=validate_each_epoch,
                    coherence_coefficient=coherence_coefficient,
                    device=device, lr=lr, latent_distribution=latent_distribution, 
                    enc_hidden_dim=enc_hidden_dim,
                    batch_size=batch_size, 
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
        config['epochs']             = self.epochs
        config['batch_size']         = self.batch_size
        config['num_enc_layers']     = self.n_encoding_layers
        config['enc_dr']             = self.enc_dr
        config['n_labels']           = self.n_labels
        config['covar_net_layers']   = 1
        if isinstance(self.latent_distribution, VonMisesDistribution):
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
        torch.save(self.model, pfile)
        config = self._get_config()
        specs = json.dumps(config, sort_keys=True, indent=4)
        with io.open(sp_file, 'w') as fp:
            fp.write(specs)
        with io.open(vocab_file, 'w') as fp:
            json.dump(self.vocabulary.get_stoi(), fp)


    def _get_wd_freqs(self, X, max_sample_size=1000000):
        sample_size = min(max_sample_size, X.shape[0])
        sums = np.array(X[:sample_size].sum(axis=0))
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
        with torch.no_grad():
            for i, ((data,labels),) in enumerate(dataloader):
                data = data.to(self.device)
                _, kl_loss, rec_loss, _ = self._forward(self.model, data)
                total_rec_loss += float(rec_loss.sum())
                total_kl_loss += float(kl_loss.sum())
        if ((total_rec_loss + total_kl_loss) / total_words) < 709.0:
            perplexity = math.exp((total_rec_loss + total_kl_loss) / total_words)
        else:
            perplexity = 1e300
        return perplexity

    def perplexity(self, X, y):
        dataloader = self._get_val_dataloader(X, y)
        self.num_val_words = X.sum()
        return self._perplexity(dataloader, self.num_val_words)


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
            with torch.no_grad():
                for i, ((data, labels),) in enumerate(val_dataloader):
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    predictions = self.model.predict(data)   ## logits of predictions
                    predictions_lists = [ p.detach().numpy() for p in list(predictions) ]
                    prediction_arrays.extend(predictions_lists)
                    if len(labels.shape) == 1:  ## standard single-label classification
                        correct = torch.argmax(predictions, dim=1) == labels
                        tot_correct += float(correct.sum())
                        tot += float((data.shape[0] - (labels < 0.0).sum())) # subtract off labels < 0 (for unlabeled data)
                    else: ## assume multilabel classification
                        correct = (torch.sigmoid(predictions) > 0.5).float() == labels
                        tot_correct += float(correct.sum())
                        tot += float(labels.nelement())
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
        val_dataloader = SingletonWrapperLoader(SparseDataLoader(val_X, val_y, batch_size=self.test_batch_size))
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
        data = data.to(self.device)
        if labels is None:
            labels = torch.zeros(data.shape[0]).unsqueeze(dim=1)
        labels = labels.to(self.device)
        
        elbo_ls, kl_ls, rec_ls, predicted_labels = \
            self._forward(self.model, data)
        if self.has_classifier:
            labels = labels.float() if self.multilabel else labels
            label_ls = self.loss_function(predicted_labels, labels).mean()
            if self.gamma < 1000.0:
                total_ls = (self.gamma * label_ls) + elbo_ls.mean()
            else:
                total_ls = label_ls + elbo_ls.mean() / self.gamma
        else:
            total_ls = elbo_ls.mean()
            label_ls = torch.zeros(total_ls.shape)
        return elbo_ls, kl_ls, rec_ls, label_ls, total_ls

    def _get_unlabeled_losses(self, model, data):
        elbo_ls, kl_ls, rec_ls, predicted_labels = \
            self._forward(model, data)
        total_ls = elbo_ls.mean() / self.gamma
        return elbo_ls, kl_ls, rec_ls, total_ls

    def fit_with_validation_loaders(self, train_dataloader, validation_dataloader, aux_dataloader,
                                    train_X_size, val_X_size, aux_X_size, total_val_words, val_X=None, val_y=None):
        
        trainer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        sc_obj, npmi, ppl, redundancy = 0.0, 0.0, 0.0, 0.0
        v_res = None
        joint_loader = PairedDataLoader(train_dataloader, aux_dataloader)
        for epoch in range(self.epochs):
            ts_epoch = time.time()
            elbo_losses = []
            lab_losses  = []
            self.model.train()
            for i, (data_batch, aux_batch) in enumerate(joint_loader):
                elbo_ls, kl_loss, _, lab_loss, total_ls = self._get_losses(self.model, data_batch)
                elbo_mean = elbo_ls.mean()
                if aux_batch is not None:
                    total_ls.backward(retain_graph=True)
                    aux_data, = aux_batch
                    aux_data, _ = aux_data # ignore (null) label
                    aux_data = aux_data.to(self.device)
                    elbo_ls_a, kl_loss_a, _, total_ls_a = self._get_unlabeled_losses(self.model, aux_data)
                    total_ls_a.backward()
                else:
                    total_ls.backward()
                trainer.step()
                trainer.zero_grad()
                if not self.quiet:
                    if aux_batch is not None:
                        elbo_losses.append(float(elbo_mean) + float(elbo_ls_a.mean()))
                    else:
                        elbo_losses.append(float(elbo_mean))                        
                    if lab_loss is not None:
                        lab_losses.append(float(lab_loss.mean()))
            if not self.quiet and not self.validate_each_epoch:
                elbo_mean = np.mean(elbo_losses) if len(elbo_losses) > 0 else 0.0
                lab_mean  = np.mean(lab_losses) if len(lab_losses) > 0 else 0.0
                self._output_status("Epoch [{}] finished in {} seconds. [elbo = {}, label loss = {}]"
                                    .format(epoch+1, (time.time()-ts_epoch), elbo_mean, lab_mean))
            if validation_dataloader is not None and (self.validate_each_epoch or epoch == self.epochs-1):
                self.model.eval()
                sc_obj, v_res = self._perform_validation(epoch, validation_dataloader, val_X_size, total_val_words, val_X, val_y)
        if v_res is None and validation_dataloader is not None:
            self.model.eval()
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
        return sc_obj, v_res


    def setup_model_with_biases(self, X: sp.csr_matrix) -> int:
        wd_freqs = self._get_wd_freqs(X)
        x_size = X.shape[0] * X.shape[1]
        if self.model is None or not self.warm_start:
            self.model = self._get_model()
            self.model.initialize_bias_terms(wd_freqs.squeeze())  ## initialize bias weights to log frequencies
        if self.npmi_matrix is not None:
            self.model.initialize_npmi_loss(self.npmi_matrix, npmi_lambda=self.npmi_lambda, npmi_scale=self.npmi_scale)
        return x_size

    

    def fit_with_validation(self, 
                            X: Union[torch.Tensor, sp.coo_matrix, sp.csr_matrix],
                            y: Union[torch.Tensor, np.ndarray],
                            val_X: Optional[Union[torch.Tensor, sp.coo_matrix, sp.csr_matrix]],
                            val_y: Optional[Union[torch.Tensor, np.ndarray]],
                            aux_X: Optional[Union[torch.Tensor, sp.coo_matrix, sp.csr_matrix]] = None,
                            opt_trial: Optional[optuna.Trial] = None) -> Tuple[float, dict]:
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

        train_dataloader = SparseDataLoader(X, y, batch_size=self.batch_size, drop_last=True)
        X_data = train_dataloader.dataset.data
        train_dataloader = SingletonWrapperLoader(train_dataloader)
        train_X_size = X_data.shape
        _ = self.setup_model_with_biases(X_data)

        if aux_X is not None:
            aux_dataloader = SparseDataLoader(X, y, batch_size=self.batch_size)
            aux_shape = aux_dataloader.dataset.data.shape
            aux_X_size = aux_shape[0] * aux_shape[1]
            aux_dataloader   = SingletonWrapperLoader(aux_dataloader)        
        else:
            aux_dataloader, aux_X_size = None, 0
        if val_X is not None:
            val_dataloader = SparseDataLoader(val_X, val_y, batch_size=self.test_batch_size)
            total_val_words = val_X.sum()
            val_X_size = val_X.shape[0] * val_X.shape[1]
            val_dataloader = SingletonWrapperLoader(val_dataloader)
        else:
            val_dataloader, total_val_words, val_X_size = None, 0, 0

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

    def _forward(self, model: BowVAEModel, data: torch.Tensor):
        """
        Forward pass of BowVAE model given the supplied data

        Parameters:
            model: Core VAE model for bag-of-words topic model
            data: Document word matrix of shape (n_train_samples, vocab_size)

        Returns:
            Tuple of:
                elbo, kl_loss, rec_loss, reconstruction
        """
        return model(data)


    def initialize_with_pretrained(self):
        assert(self.pretrained_param_file is not None)
        self.model = self._get_model()
        #self.model.load_parameters(self.pretrained_param_file, allow_missing=False)


    def _get_model(self):
        """
        Initializes embedding weights and returns a `BowVAEModel` with hyperparameters provided.

        Returns:
            (:class:`BowVAEModel`) initialized using provided hyperparameters
        """
        #vocab, emb_size = self._initialize_embedding_layer(self.embedding_source, self.embedding_size)
        emb_size = self.embedding_size
        model = \
                BowVAEModel(self.enc_hidden_dim, emb_size, n_encoding_layers=self.n_encoding_layers,
                            vocab_size=len(self.vocabulary),
                            enc_dr=self.enc_dr, 
                            classifier_dropout=self.classifier_dropout,
                            n_labels = self.n_labels,
                            gamma = self.gamma,
                            multilabel = self.multilabel,
                            latent_distribution=self.latent_distribution, 
                            device=self.device)
        if self.pretrained_param_file is not None:
            model = torch.load(self.pretrained_param_file)
        model.to(self.device)
        return model
    

    def get_topic_vectors(self) -> torch.Tensor:
        """
        Get topic vectors of the fitted model.

        Returns:
            topic_distribution: topic_distribution[i, j] represents word j in topic i. shape=(n_latent, vocab_size)
        """

        return self.model.get_topic_vectors() 

    def transform(self, X: sp.csr.csr_matrix) -> torch.Tensor:
        """
        Transform data X according to the fitted model.

        Parameters:
            X: Document word matrix of shape {n_samples, n_features}

        Returns:
            topic_distribution: shape=(n_samples, n_latent) Document topic distribution for X
        """
        mx_array = mx.nd.array(X,dtype='float32')
        return self.model.encode_data(mx_array).detach().numpy()


class BowMetricEstimator(BowEstimator):

    def __init__(self, *args, sdml_smoothing_factor=0.3, non_scoring_index=-1, **kwargs):
        super(BowMetricEstimator, self).__init__(*args, **kwargs)
        self.loss_function = GeneralizedSDMLLoss(smoothing_parameter=sdml_smoothing_factor)
        self.non_scoring_index = non_scoring_index


    @classmethod
    def from_config(cls, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        return est

    def _get_model(self, bow_size=-1):
        if self.embedding_source != 'random':
            e_type, e_name = tuple(self.embedding_source.split(':'))
            #pt_embedding = nlp.embedding.create(e_type, source=e_name)
            #self.vocabulary.set_embedding(pt_embedding)
            #emb_size = len(self.vocabulary.embedding.idx_to_vec[0])
            #for word in self.vocabulary.embedding._idx_to_token:
            #    if (self.vocabulary.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
            #        self.vocabulary.embedding[word] = mx.nd.random.normal(0, 0.1, emb_size)
        else:
            emb_size = self.embedding_size
        model = \
            MetricBowVAEModel(self.enc_hidden_dim, emb_size, n_encoding_layers=self.n_encoding_layers,
                            enc_dr=self.enc_dr, fixed_embedding=self.fixed_embedding,
                            classifier_dropout=self.classifier_dropout,
                            n_labels = self.n_labels,
                            gamma = self.gamma,
                            multilabel = self.multilabel,
                            latent_distribution=self.latent_distribution, 
                            device=self.device)
        if self.pretrained_param_file is not None:
            model.load_parameters(self.pretrained_param_file, allow_missing=False)
        return model
        

    def _get_model_bias_initialize(self, train_data):
        model = self._get_model()
        tr_bow_matrix = self._get_bow_matrix(train_data)
        model.initialize_bias_terms(tr_bow_matrix.sum(axis=0))
        if self.npmi_matrix is not None:
            self.model.initialize_npmi_loss(self.npmi_matrix)
        return model

    def _forward(self, model, data):
        elbo_ls, rec_ls, kl_ls, red_ls, total_ls = self._get_unlabeled_losses(model, data)
        return elbo_ls, rec_ls, kl_ls, red_ls, total_ls, None

    def _ff_batch(self, model, batch_data):
        (batch1, labels1), (batch2, labels2) = batch_data
        batch1 = batch1.to(self.device)
        batch2 = batch2.to(self.device)
        labels1 = labels1.to(self.device)
        labels2 = labels2.to(self.device)
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
            label_mat = self.loss_function._compute_labels(label1, label2)
            dists = self.loss_function._compute_distances(z_mu1, z_mu2)
            probs = torch.nn.functional.softmax(-dists, axis=1).detach().numpy()
            posteriors += list(probs)
            label1 = np.array(label1.squeeze().detach().numpy(), dtype='int')
            ground_truth_idx += list(label1) ## index values for labels
            gt = np.zeros((label1.size()[0], int(max(label2).asscalar())+1))
            gt[np.arange(label1.size()[0]), label1] = 1
            ground_truth += list(gt)
            if emb2 is None:
                emb2 = z_mu2.detach().numpy()
            emb1 += list(z_mu1.detach().numpy())
        posteriors = np.array(posteriors)
        ground_truth = np.array(ground_truth)
        ground_truth_idx = np.array(ground_truth_idx)
        labels = np.arange(posteriors[0].size()[0])        
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
        return v_res['avg_prec'], v_res


class SeqBowEstimator(BaseEstimator):

    def __init__(self, *args,
                 llm_model_name = 'distilbert-base-uncased',
                 n_labels = 0,
                 log_interval=5,
                 warmup_ratio=0.1,
                 gamma=1.0,
                 multilabel=False,
                 decoder_lr = 0.01,
                 checkpoint_dir = None,
                 classifier_dropout = 0.0,
                 pure_classifier_objective = False,
                 validate_each_epoch = False,
                 entropy_loss_coef = 0.0,
                 pool_encoder = True,
                 **kwargs):
        super(SeqBowEstimator, self).__init__(*args, **kwargs)
        self.pure_classifier_objective = pure_classifier_objective
        self.validate_each_epoch = validate_each_epoch
        self.minimum_lr = 1e-9
        self.checkpoint_dir = checkpoint_dir
        self.llm_model_name = llm_model_name
        self.has_classifier = n_labels >= 2
        self.classifier_dropout = classifier_dropout
        self.multilabel = multilabel
        self.n_labels = n_labels
        self.metric = None if n_labels == 0 else (MultilabelAUPRC(num_classes=n_labels) if multilabel else MulticlassAUPRC(num_classes=n_labels))
        self.warmup_ratio = warmup_ratio
        self.log_interval = log_interval
        self.loss_function = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
        self.gamma = gamma
        self.decoder_lr = decoder_lr
        self._bow_matrix = None
        self.entropy_loss_coef = entropy_loss_coef
        self.pool_encoder = pool_encoder


    @classmethod
    def from_config(cls,
                    config: Union[str, dict],
                    vocabulary: torchtext.vocab.Vocab,
                    log_interval: int = 1,
                    pretrained_param_file: Optional[str] = None,
                    n_labels: Optional[int] = None,                    
                    device: str = 'cpu') -> 'SeqBowEstimator':
        """
        Instantiate an object of this class using the provided `config`

        Parameters:
            config: String to configuration path (in json format) or an autogluon dictionary representing the config
            log_interval: Logging frequency (default = 1)
            pretrained_param_file: Parameter file
            device: pytorch device
        
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
        ldist_def = config['latent_distribution']
        llm_model_name = config['llm_model_name']
        model = torch.load(pretrained_param_file, map_location=device)

        latent_distribution = model.latent_distribution
        estimator = cls(llm_model_name = llm_model_name,
                        vocabulary = vocabulary,
                        n_labels        = config.get('n_labels', n_labels),
                        latent_distribution = latent_distribution,
                        batch_size      = int(config['batch_size']),
                        warmup_ratio = config['warmup_ratio'],
                        classifier_dropout = config['classifier_dropout'],
                        epochs = int(config['epochs']),
                        gamma = config['gamma'],
                        lr = config['lr'],
                        decoder_lr = config['decoder_lr'],
                        pretrained_param_file = pretrained_param_file,
                        warm_start = (pretrained_param_file is not None),
                        log_interval=log_interval,
                        device=device)
        estimator.model = model
        estimator.model.device = device
        return estimator

    @classmethod
    def from_saved(cls, model_dir: str,
                   log_interval: int = 1,
                   device: Optional[str] = 'cpu') -> 'SeqBowEstimator':
        if model_dir is not None:
            param_file = os.path.join(model_dir, 'model.params')
            vocab_file = os.path.join(model_dir, 'vocab.bin')
            config_file = os.path.join(model_dir, 'model.config')
        with open(config_file) as f:
            config = json.loads(f.read())
        vocab = torch.load(vocab_file)
        return cls.from_config(config,
                               vocabulary = vocab,
                               log_interval = log_interval,
                               pretrained_param_file = param_file,
                               device = device)
    

    def _get_model_bias_initialize(self, train_data):
        model = self._get_model()
        tr_bow_counts = self._get_bow_wd_counts(train_data)
        model.initialize_bias_terms(tr_bow_counts)
        if self.npmi_matrix is not None:
            print("****** INITIALIZING NPMI LOSS FUNCTION *******")
            model.initialize_npmi_loss(self.npmi_matrix)
        return model
        
    
    def _get_model(self):
        llm_base_model = get_llm_model(self.llm_model_name).to(self.device)
        model = SeqBowVED(llm_base_model, self.latent_distribution, num_classes=self.n_labels, device=self.device, 
                          vocab_size = len(self.vocabulary), use_pooling = self.pool_encoder,
                          entropy_loss_coef=self.entropy_loss_coef,
                          dropout=self.classifier_dropout)
        return model

    def _get_config(self):
        config = {}
        config['lr'] = self.lr
        config['decoder_lr'] = self.decoder_lr
        config['n_latent'] = self.n_latent
        config['n_labels'] = self.n_labels
        config['batch_size'] = self.batch_size
        if isinstance(self.latent_distribution, VonMisesDistribution):
            config['latent_distribution'] = {'dist_type':'vmf', 'kappa': self.latent_distribution.kappa}
        elif isinstance(self.latent_distribution, LogisticGaussianDistribution):
            config['latent_distribution'] = {'dist_type':'logistic_gaussian', 'alpha':self.latent_distribution.alpha}
        else:
            config['latent_distribution'] = {'dist_type':'gaussian'}
        config['epochs'] = self.epochs
        #config['embedding_source'] = self.embedding_source
        config['gamma'] = self.gamma
        config['warmup_ratio'] = self.warmup_ratio
        config['llm_model_name'] = self.llm_model_name
        config['classifier_dropout'] = self.classifier_dropout
        return config

    def write_model(self, model_dir: str, suffix: str ='', vectorizer=None) -> None:
        """
        Writes the model within this estimator to disk.

        Parameters:
            model_dir: Output directory for model parameters, config and vocabulary
            suffix: Suffix to use for model (e.g. at different checkpoints)
        """
        pfile = os.path.join(model_dir, ('model.params' + suffix))
        conf_file = os.path.join(model_dir, ('model.config' + suffix))
        vocab_file = os.path.join(model_dir, ('vocab.bin' + suffix))
        torch.save(self.model, pfile)
        config = self._get_config()
        specs = json.dumps(config, sort_keys=True, indent=4)
        if vectorizer is not None:
            vectorizer_file = os.path.join(model_dir, ('vectorizer.pkl' + suffix))
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(vectorizer, f)
        with open(conf_file, 'w') as f:
            f.write(specs)
        torch.save(self.vocabulary, vocab_file)


    def log_train(self, batch_id, batch_num, step_loss, rec_loss, red_loss, class_loss,
                  log_interval, epoch_id, learning_rate):
        """Generate and print out the log message for training. """
        if self.has_classifier:
            #metric_nm, metric_val = self.metric.compute()
            #if not isinstance(metric_nm, list):
            #    metric_nm, metric_val = [metric_nm], [metric_val]
            metric_nm = "AUPRC"
            try:
                metric_val = self.metric.compute()
            except:
                metric_val = 0.0
            self._output_status("Epoch {} Batch {}/{} loss={}, (rec_loss = {}), (red_loss = {}), (class_loss = {}) lr={:.10f}, metrics[{}]: {}"
                                .format(epoch_id+1, batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval, red_loss/log_interval,
                                        class_loss/log_interval, learning_rate, metric_nm, metric_val))
        else:
            self._output_status("Epoch {} Batch {}/{} loss={}, (rec_loss = {}), (red_loss = {}), (class_loss = {}) lr={:.10f}"
                                .format(epoch_id+1, batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval, red_loss/log_interval,
                                        class_loss/log_interval, learning_rate))

    def log_eval(self, batch_id, batch_num, step_loss, rec_loss, log_interval):
        if self.metric is not None:
            metric_val = self.metric.compute()
            metric_nm = 'AuPRC'
            if not isinstance(metric_nm, list):
                metric_nm, metric_val = [metric_nm], [metric_val]
            self._output_status("Batch {}/{} loss={} (rec_loss = {}), metrics: {:.10f}"
                .format(batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval, *metric_val))
        else:
            self._output_status("Batch {}/{} loss={} (rec_loss = {}), metrics: NA"
              .format(batch_id+1, batch_num, step_loss/log_interval, rec_loss/log_interval))

    def _get_bow_matrix(self, dataloader, cache=False):
        bow_matrix = []
        max_rows = 2000000000 / len(self.vocabulary)
        logging.info("Maximum rows for BOW matrix = {}".format(max_rows))
        rows = 0
        for i, data in enumerate(dataloader):
            seqs, = data
            #bow_batch = list(seqs[3].squeeze(axis=1))
            bow_batch = list(seqs[3])
            rows += len(bow_batch)
            if i >= max_rows:
                break
            bow_matrix.extend(bow_batch)
        bow_matrix = torch.vstack(bow_matrix)
        if cache:
            self._bow_matrix = bow_matrix
        return bow_matrix

    def _get_bow_wd_counts(self, dataloader):
        sums = torch.zeros(len(self.vocabulary)).to(self.device)
        for i, data in enumerate(dataloader):
            seqs, = data
            bow_batch = seqs[3].to_dense()
            sums += bow_batch.sum(axis=0)
        return sums.cpu().numpy()

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
        label, input_ids, mask, bow = seqs
        elbo_ls, rec_ls, kl_ls, red_ls, out = model(input_ids.to(self.device), mask.to(self.device), bow.to(self.device))
        if self.has_classifier:
            label = label.to(self.device)
            label_ls = self.loss_function(out, label)
            label_ls = label_ls.mean()
            total_ls = (self.gamma * label_ls) + elbo_ls.mean()
            if not self.multilabel:
                #label_ind = label.argmax(dim=0)
                #self.metric.update([out], [label_ind])
                self.metric.update(torch.tensor(out), torch.tensor(label))
                #self.metric.update(torch.Tensor([out]), torch.Tensor([label_ind]))
            else:
                self.metric.update([out], [label])
        else:
            total_ls = elbo_ls.mean()
            label_ls = torch.zeros(total_ls.size())        
        return elbo_ls, rec_ls, kl_ls, red_ls.mean(), label_ls, total_ls

    def _get_unlabeled_losses(self, model, batch_data):
        seqs, = batch_data
        _ , input_ids, mask, bow = seqs
        elbo_ls, rec_ls, kl_ls, red_ls, out = model(
            input_ids.to(self.device), mask.to(self.device), bow.to(self.device))
        total_ls = elbo_ls.mean() / self.gamma
        return elbo_ls, rec_ls, kl_ls, red_ls.mean(), total_ls
        

    def fit_with_validation(self,
                            train_data: torch.utils.data.DataLoader,
                            dev_data: torch.utils.data.DataLoader,
                            aux_data: torch.utils.data.DataLoader):
        """
        Training function.

        Parameters:
            train_data: Dataloader with training data.
            dev_data: Dataloader with dev/validation data.
            aux_data: Dataloader with auxilliary data.
        """
        if self.model is None or not self.warm_start:
            self.model = self._get_model_bias_initialize(train_data)

        model = self.model

        accumulate = False
        v_res      = None


        joint_loader = PairedDataLoader(train_data, aux_data)
        num_train_steps = len(joint_loader) * self.epochs

        ## The following from HuggingFace trainer.py lines 1047 to 1063
        decay_parameters = get_parameter_names(model.llm, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        non_llm_parameters = [name for name,_ in model.named_parameters() if not name.startswith("llm")]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.llm.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 1e-3,
            },
            { "params": [
                p for n, p in model.llm.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
              "weight_decay": 0.0
             }
        ]
        dec_optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in non_llm_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.00001,
                },
            ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr = self.lr, eps=1e-6, betas=(0.9, 0.999))
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

        dec_optimizer = torch.optim.Adam(dec_optimizer_grouped_parameters,
                                         lr = self.decoder_lr,
                                         eps = 1e-6,
                                         weight_decay = 0.00001)
        
        if accumulate:
            num_train_steps /= accumulate
        
        logging.info("Number of warmup steps = {}, num total train steps = {}, batch_size = {}, epochs = {}"
                     .format(num_warmup_steps, num_train_steps, self.batch_size, self.epochs))
        step_num = 0

        loss_details = { 'step_loss': 0.0, 'elbo_loss': 0.0, 'red_loss': 0.0, 'class_loss': 0.0 }
        def update_loss_details(total_ls, elbo_ls, red_ls, class_ls):
            loss_details['step_loss'] += float(total_ls.mean())
            loss_details['elbo_loss'] += float(elbo_ls.mean())
            loss_details['red_loss'] += float(red_ls.mean())
            if class_ls is not None:
                loss_details['class_loss'] += float(class_ls.mean())
            
        sc_obj = None
        v_res  = None
        
        for epoch_id in range(self.epochs):
            if self.metric is not None:
                self.metric.reset()
            model.train()
            model.llm.train()
            
            for (batch_id, (data, aux_batch)) in enumerate(joint_loader):
                # data_batch is either a 2-tuple of: (labeled, unlabeled)
                # OR a 1-tuple of (labeled,)
                
                # forward and backward with optional auxilliary data
                elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls = self._get_losses(model, data)
                if aux_batch is not None:
                    total_ls.backward(retain_graph=True)
                    elbo_ls_2, rec_ls_2, kl_ls_2, red_ls_2, total_ls_2 = self._get_unlabeled_losses(model, aux_batch)
                    total_ls_2.backward()
                else:
                    total_ls.backward()
                update_loss_details(total_ls, elbo_ls, red_ls, label_ls)
                if aux_batch is not None:
                    update_loss_details(total_ls_2, elbo_ls_2, red_ls_2, None)
                    
                #debug

                if not accumulate or (batch_id + 1) % accumulate == 0:
                    #torch.nn.utils.clip_grad.clip_grad_value_(model.llm.parameters(), 1.0)
                    optimizer.step()
                    dec_optimizer.step()
                    lr_scheduler.step()
                    model.zero_grad()
                    step_num += 1
                if (batch_id + 1) % (self.log_interval) == 0:
                    lr = lr_scheduler.get_last_lr()[0] # get lr from first group
                    self.log_train(batch_id, num_train_steps // self.epochs, loss_details['step_loss'],
                                       loss_details['elbo_loss'], loss_details['red_loss'], loss_details['class_loss'], self.log_interval,
                                       epoch_id, lr)
                    ## reset loss details
                    for d in loss_details:
                        loss_details[d] = 0.0
            # inference on dev data
            if dev_data is not None and (self.validate_each_epoch or epoch_id == (self.epochs-1)):
                sc_obj, v_res = self._perform_validation(model, dev_data, epoch_id)
            else:
                sc_obj, v_res = None, None
            if self.checkpoint_dir:
                self.write_model(self.checkpoint_dir, suffix=str(epoch_id))
        if v_res is None and dev_data is not None:
            sc_obj, v_res = self._perform_validation(model, dev_data, 0)
        return sc_obj, v_res


    def _compute_coherence(self, model, k, test_data, log_terms=False):
        num_topics = model.n_latent
        sorted_ids = model.get_ordered_terms()
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
        return npmi, redundancy
    

    def _perform_validation(self, model, dev_data, epoch_id):
        model.eval()
        v_res, metric_nm, metric_val = self.validate(model, dev_data)
        sc_obj = self._get_objective_from_validation_result(v_res)
        if 'accuracy' in v_res:
            self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}. Accuracy = {}."
                                .format(epoch_id, sc_obj, v_res['ppl'], v_res['npmi'], v_res['redundancy'], v_res['accuracy']))
        else:
            self._output_status("Epoch [{}]. Objective = {} ==> PPL = {}. NPMI ={}. Redundancy = {}."
                                .format(epoch_id, sc_obj, v_res['ppl'], v_res['npmi'], v_res['redundancy']))
        return sc_obj, v_res
                
    
    def validate(self, model, dataloader):
        bow_matrix = self._bow_matrix if self._bow_matrix is not None else self._get_bow_matrix(dataloader, cache=True)
        num_words = torch.sparse.sum(bow_matrix)
        npmi, redundancy = self._compute_coherence(model, 10, bow_matrix, log_terms=True)
        if self.metric is not None:
            self.metric.reset()
        step_loss = 0
        elbo_loss  = 0
        total_rec_loss = 0.0
        total_kl_loss  = 0.0
        model.eval()
        for batch_id, seqs in enumerate(dataloader):
            elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls = self._get_losses(model, seqs)
            total_rec_loss += float(rec_ls.sum().cpu().detach())
            total_kl_loss  += float(kl_ls.sum().cpu().detach())
            step_loss += float(total_ls.mean().cpu().detach())
            elbo_loss  += float(elbo_ls.mean().cpu().detach())
            if (batch_id + 1) % (self.log_interval) == 0:
                logging.debug('All loss terms: {}, {}, {}, {}, {}, {}'.format(elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls))
                self.log_eval(batch_id, len(dataloader), step_loss, elbo_loss, self.log_interval)
                step_loss = 0
                elbo_loss = 0
        likelihood = (total_rec_loss + total_kl_loss) / float(num_words)
        if likelihood < 709.0:
            perplexity = math.exp(likelihood)
        else:
            perplexity = 1e300
        v_res = {'ppl':perplexity, 'npmi': npmi, 'redundancy': redundancy}
        metric_nm = 0.0
        metric_val = 0.0
        if self.has_classifier:
            metric_val = self.metric.compute()
            metric_nm = 'AuPRC'
            if not isinstance(metric_nm, list):
                metric_nm, metric_val = [metric_nm], [metric_val]
            self._output_status("Validation metric: {:.6}".format(metric_val[0]))
            v_res['accuracy'] = metric_val[0]
        return v_res, metric_nm, metric_val


class SeqBowMetricEstimator(SeqBowEstimator):

    def __init__(self, *args, sdml_smoothing_factor=0.3, metric_loss_temp=0.1, use_teacher_forcing=False, 
                 teacher_forcing_mode='rand',
                 use_sdml=False, non_scoring_index=-1, **kwargs):
        super(SeqBowMetricEstimator, self).__init__(*args, **kwargs)
        if use_teacher_forcing:
            self.loss_function = CrossBatchCosineSimilarityLoss(teacher_mode = teacher_forcing_mode)
        else:
            self.loss_function = \
                GeneralizedSDMLLoss(smoothing_parameter=sdml_smoothing_factor, x2_downweight_idx=non_scoring_index) if use_sdml \
                else MultiNegativeCrossEntropyLoss(smoothing_parameter=sdml_smoothing_factor, metric_loss_temp=metric_loss_temp)
        self.non_scoring_index = non_scoring_index ## if >=0 this will avoid considering this label index in evaluation


    @classmethod
    def from_config(cls, *args, **kwargs):
        est = super().from_config(*args, **kwargs)
        return est
    

    def _get_model(self):
        llm_base_model = get_llm_model(self.llm_model_name).to(self.device)
        model = MetricSeqBowVED(llm_base_model, self.latent_distribution, num_classes=self.n_labels, device=self.device, 
                                vocab_size = len(self.vocabulary), use_pooling=self.pool_encoder,
                                dropout=self.classifier_dropout, entropy_loss_coef=self.entropy_loss_coef)
        return model
        
    def _get_bow_wd_counts(self, dataloader):
        sums = torch.zeros(len(self.vocabulary)).to(self.device)
        for i, (data_a, data_b) in enumerate(dataloader):
            seqs_a = data_a
            bow_batch_a = seqs_a[3].to_dense()
            seqs_b = data_b
            bow_batch_b = seqs_b[3].to_dense()
            sums += bow_batch_a.sum(axis=0)
            sums += bow_batch_b.sum(axis=0)
        return sums.cpu().numpy()        
        
    def _get_bow_matrix(self, dataloader, cache=False):
        bow_matrix = []
        for _, seqs in enumerate(dataloader):
            batch_1, batch_2 = seqs                
            bow_matrix.extend(list(batch_2[3].to_dense().squeeze(axis=1)))
            bow_matrix.extend(list(batch_1[3].to_dense().squeeze(axis=1)))
        bow_matrix = torch.vstack(bow_matrix)
        if cache:
            self._bow_matrix = bow_matrix
        return bow_matrix

    def _ff_batch(self, model, batch_data):
        batch1, batch2 = batch_data
        label1, in1, mask1, bow1 = batch1
        label2, in2, mask2, bow2 = batch2
        elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2 = model(
            in1.to(self.device), mask1.to(self.device), bow1.to(self.device),
            in2.to(self.device), mask2.to(self.device), bow2.to(self.device))
        return elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2

    def _get_losses(self, model, batch_data):
        elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1, label2 = self._ff_batch(model, batch_data)
        ## convert back to label indices rather than 1-hot vecs
        label1 = label1.to(self.device)
        label2 = label2.to(self.device)
        label_ls = self.loss_function(z_mu1, label1, z_mu2, label2)
        total_ls = (label_ls) + (elbo_ls.sum() / self.gamma)   # .mean()
        return elbo_ls, rec_ls, kl_ls, red_ls, label_ls, total_ls

    def _get_unlabeled_losses(self, model, batch_data):
        batch1, = batch_data
        _, input_ids1, mask1, bow1 = batch1
        elbo_ls, rec_ls, kl_ls, red_ls = model.unpaired_input_forward(
            input_ids1.to(self.device), mask1.to(self.device), bow1.to(self.device))
        total_ls = elbo_ls.mean() / self.gamma
        return elbo_ls, rec_ls, kl_ls, red_ls, total_ls
    
    def validate(self, model, dataloader, epoch_id, include_predictions=True):
        kl_ls_tot = 0.0
        elbo_ls_tot = 0.0
        for _, data_batch in enumerate(dataloader):
            elbo_ls, rec_ls, kl_ls, red_ls, z_mu1, z_mu2, label1_ind, label2_ind = self._ff_batch(model, data_batch)
            elbo_ls_tot += float(elbo_ls.sum())
            kl_ls_tot   += float(kl_ls.sum())

        return {'elbo_ls': elbo_ls_tot, 'kl_ls': kl_ls_tot}
                

            
    def _perform_validation(self, model, dev_data, epoch_id):
        v_res = self.validate(model, dev_data, epoch_id)
        self._output_status("Epoch [{}]. ==> elbo loss = {}; kldiv loss = {}"
                            .format(epoch_id, v_res['elbo_ls'], v_res['kl_ls']))
        return v_res['kl_ls'], v_res

