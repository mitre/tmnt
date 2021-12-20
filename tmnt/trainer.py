# coding: utf-8
# Copyright (c) 2020-2021 The MITRE Corporation.
"""
Model trainers that handle data prep, pre-trained vocabularies and enable model selection.
"""

import json
import os
import logging
import mxnet as mx
import numpy as np
import gluonnlp as nlp
import datetime
import time
import copy
import statistics
import autogluon.core as ag

from autogluon.core.scheduler.reporter import FakeReporter
from pathlib import Path
from tmnt.utils import log_utils
from tmnt.utils.random import seed_rng
from tmnt.utils.log_utils import logging_config
from tmnt.data_loading import load_vocab, file_to_data
from tmnt.bert_handling import get_bert_datasets, JsonlDataset
from tmnt.estimator import BowEstimator, CovariateBowEstimator, SeqBowEstimator
from tmnt.preprocess.vectorizer import TMNTVectorizer
from mxnet.gluon.data import ArrayDataset


class BaseTrainer(object):
    """Abstract base class for topic model trainers. 

    Objects of this class provide all functionality for training a topic model, including 
    handle data prep/loading, model definition, and training parameters. Trainer objects include
    a tmnt.estimator.BaseEstimator that is responsible for fitting/estimating the topic model parameters.

    Parameters:

        train_data (array-like or sparse matrix): Training input data tensor
        test_data (array-like or sparse matrix): Testing/validation input data tensor
    """

    def __init__(self, model_out_dir=None, train_data_or_path=None, test_data_or_path=None, aux_data_or_path=None, use_gpu=False, val_each_epoch=False, rng_seed=1234):
        self.model_out_dir = model_out_dir
        self.train_data_or_path   = train_data_or_path
        self.test_data_or_path    = test_data_or_path
        self.aux_data_or_path     = aux_data_or_path
        self.use_gpu      = use_gpu
        self.rng_seed     = rng_seed        
        self.vocab_cache  = {}
        self.validate_each_epoch = val_each_epoch


    def _get_x_y_data(self, data_source):
        if isinstance(data_source, str):
            X, y, _, _ = file_to_data(data_source, len(self.vocabulary))
        elif isinstance(data_source, tuple):
            X, y = data_source
        else:
            X, y = data_source, None
        return X, y
        

    def x_get_mxnet_visible_gpus(self):
        """Gets visible gpus from MXNet.
        
        Returns:
            (:class:`mxnet.context.Context`): representing the GPU context
        """
        gpu_count = 0
        while True:
            try:
                arr = mx.np.array(1.0, ctx=mx.gpu(gpu_count))
                arr.asnumpy()
                gpu_count += 1
            except Exception:
                break
        return [mx.gpu(i) for i in range(gpu_count)]

    def _get_mxnet_visible_gpus(self):
        ln = 0
        t = datetime.datetime.now()
        while ln < 1 and ((datetime.datetime.now() - t).seconds < 30):
            time.sleep(1)
            gpus = self.x_get_mxnet_visible_gpus()
            ln = len(gpus)
        if ln > 0:
            return gpus
        else:
            raise Exception("Unable to get a gpu after 30 tries")
        

    def train_with_single_config(self, config, num_evals):
        raise NotImplementedError()

    def train_model(self, config, reporter):
        """Train a model with config and reporter callback.
        """
        raise NotImplementedError()


class TopicTrainer(BaseTrainer):
    """
        vocabulary (`gluonnlp.Vocab`): Gluon NLP vocabulary object representing the bag-of-words used for the dataset
    """

    def __init__(self, vocabulary, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocabulary   = vocabulary
        
    
    def _initialize_vocabulary(self, embedding_source, set_vocab=True):
        """Initialize the embedding layer randomly or using pre-trained embeddings provided
        
        Args:
            embedding_source (str): denoting a Gluon-NLP embedding source with the format <type>:<name> where <type>
                is 'glove', 'fasttext' or 'word2vec' and <name> denotes the embedding name (e.g. 'glove.6B.300d').
                See `gluonnlp.embedding.list_sources()` for a full list
            set_vocab (bool): actually instantiate vocabulary (as opposed to just caching, when set_vocab=False)

        Returns:
            (tuple): tuple containing:
                vocab (`gluonnlp.Vocab`): Resulting GluonNLP vocabulary with initialized embedding
                emb_size (int): Size of embedding (based on pre-trained embedding, random returns -1 for size to be set later)
        """
        vocab = self.vocabulary
        if embedding_source != 'random':
            if self.vocab_cache.get(embedding_source):
                pt_embedding = copy.deepcopy(self.vocab_cache[embedding_source])
            else:
                e_type, e_name = tuple(embedding_source.split(':'))
                if e_type == 'file':
                    if not os.path.exists(e_name):
                        raise Exception("Embedding file not found: {}".format(e_name))
                    pt_embedding = nlp.embedding.TokenEmbedding.from_file(e_name)
                else:
                    pt_embedding = nlp.embedding.create(e_type, source=e_name)
                self.vocab_cache[embedding_source] = copy.deepcopy(pt_embedding) ## cache another copy 
            emb_size = len(pt_embedding.idx_to_vec[0])
            if set_vocab:
                vocab.set_embedding(pt_embedding)
                num_oov = 0
                for word in vocab.embedding._idx_to_token:
                    if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                        logging.debug("Term {} is OOV".format(word))
                        num_oov += 1
                        vocab.embedding[word] = mx.nd.random.normal(0, 0.1, emb_size)
                logging.debug(">> {} Words did not appear in embedding source {}".format(num_oov, embedding_source))
        else:
            vocab.set_embedding(None) ## unset embedding
            emb_size = -1
        return vocab, emb_size

    

    def train_with_single_config(self, config, num_evals):
        """Fit models with a single configuration and report the value of the objective function.

        This method trains a model defined by the configuration `num_evals` times. Each time
        the model weights are randomly initialized with a different RNG seed. The results
        of each run are captured and mean and std reported.
        
        Args:
            config (dict): Configuration instance with hyperparameter values for model definition.
            num_evals (int): Number of model fits and evaluations to perform (with random initialization)

        Returns:
            (tuple): Tuple containing:
                - model (:class:`tmnt.modeling.BowVAEModel`): VAE Model instance with trained/fit parameters.
                - obj (float): objective value of the objective function with the best model.
       """
        rng_seed = self.rng_seed
        best_obj = -1000000000.0
        best_model = None
        best_vres = None
        if self.test_data_or_path is not None:
            #if c_args.tst_vec_file:
            #    trainer.set_heldout_data_path_as_test()        
            logging.info("Training with config: {}".format(config))
            npmis, perplexities, redundancies, objectives = [],[],[],[]
            ntimes = int(num_evals)
            for i in range(ntimes):
                seed_rng(rng_seed) # update RNG
                logging.info("Setting rng seed to {}".format(rng_seed))
                rng_seed += 1
                model, obj, v_res = self.train_model(config, FakeReporter())
                npmis.append(v_res['npmi'])
                perplexities.append(v_res['ppl'])
                redundancies.append(v_res['redundancy'])
                objectives.append(obj)
                if obj > best_obj:
                    best_vres = v_res
                    best_obj = obj
                    best_model = model
            #test_type = "HELDOUT" if c_args.tst_vec_file else "VALIDATION"
            test_type = "VALIDATION"
            if ntimes > 1:
                logging.info("Final {} NPMI         ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(npmis), statistics.stdev(npmis)))
                logging.info("Final {} Perplexity   ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(perplexities), statistics.stdev(perplexities)))
                logging.info("Final {} Redundancy   ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(redundancies), statistics.stdev(redundancies)))
                logging.info("Final {} Objective    ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(objectives), statistics.stdev(objectives)))
            else:
                logging.info("Final {} NPMI           ==> {}".format(test_type, npmis[0]))
                logging.info("Final {} Perplexity     ==> {}".format(test_type, perplexities[0]))
                logging.info("Final {} Redundancy     ==> {}".format(test_type, redundancies[0]))
                logging.info("Final {} Objective      ==> {}".format(test_type, objectives[0]))            
            return best_model, best_obj, best_vres
        else:
            return self.train_model(config, FakeReporter())
    


class BowVAETrainer(TopicTrainer):
    """Trainer for bag-of-words VAE topic models.
    
    Parameters:
        vocabulary (`gluonnlp.Vocab`): Gluon NLP vocabulary object representing the bag-of-words used for the dataset
        log_out_dir (str): String path to directory for outputting log file and potentially saved model information
        model_out_dir (str): Explicit string path to saved model information
        train_data_or_path (`scipy.sparse.csr.csr_matrix`|str): Training input path
        test_data_or_path (`scipy.sparse.csr.csr_matrix`|str): Testing/validation input path
        use_gpu (bool): Flag to force use of a GPU if available.  Default = False.
        val_each_epoch (bool): Perform validation (NPMI and perplexity) on the validation set after each epoch. Default = False.
        rng_seed (int): Seed for random number generator. Default = 1234
    """
    def __init__(self, vocabulary, train_data_or_path, test_data_or_path,
                 log_out_dir='_exps', model_out_dir='_model_dir', coherence_via_encoder=False, aux_data_or_path=None,
                 pretrained_param_file=None, topic_seed_file = None, use_labels_as_covars=False, coherence_coefficient=8.0,
                 use_gpu=False, n_labels=0,
                 val_each_epoch=True, rng_seed=1234):
        super().__init__(vocabulary, model_out_dir, train_data_or_path, test_data_or_path, aux_data_or_path, use_gpu, val_each_epoch, rng_seed)
        if not log_utils.CONFIGURED:
            logging_config(folder=log_out_dir, name='tmnt', level='info', console_level='info')
        self.log_out_dir = log_out_dir
        self.seed_matrix = None
        self.pretrained_param_file = pretrained_param_file
        self.n_labels = n_labels
        self.use_labels_as_covars = use_labels_as_covars
        self.coherence_via_encoder = coherence_via_encoder
        self.coherence_coefficient = coherence_coefficient
        if topic_seed_file:
            self.seed_matrix = get_seed_matrix_from_file(topic_seed_file, vocabulary, ctx)
        
    @classmethod
    def from_arguments(cls, c_args, val_each_epoch=True):
        """Constructor method to build BowVAETrainer from command-line arguments directly.
        
        Parameters:
            c_args (`argparse.Namespace`): Command-line arguments.
            val_each_epoch (bool): Flag for performing validation each epoch. optional (default = True)
        """
        i_dt = datetime.datetime.now()
        log_out_dir = \
            os.path.join(c_args.save_dir,
                         "train_{}_{}_{}_{}_{}_{}_{}"
                         .format(i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second,i_dt.microsecond))
        if not os.path.exists(log_out_dir):
            lpath = Path(log_out_dir)
            lpath.mkdir(parents=True, exist_ok=True)
        if not log_utils.CONFIGURED:
            logging_config(folder=log_out_dir, name='tmnt', level=c_args.log_level, console_level=c_args.log_level)
        logging.info(c_args)
        seed_rng(c_args.seed)
        if c_args.vocab_file and c_args.tr_vec_file:
            vpath = Path(c_args.vocab_file)
            tpath = Path(c_args.tr_vec_file)
            if not (vpath.is_file() and tpath.is_file()):
                raise Exception("Vocab file {} and/or training vector file {} do not exist"
                                .format(c_args.vocab_file, c_args.tr_vec_file))
        logging.info("Loading data via pre-computed vocabulary and sparse vector format document representation")
        vocab = load_vocab(c_args.vocab_file, encoding=c_args.str_encoding)
        voc_size = len(vocab)
        X, y, wd_freqs, _ = file_to_data(c_args.tr_vec_file, voc_size)
        model_out_dir = c_args.model_dir if c_args.model_dir else os.path.join(log_out_dir, 'MODEL')
        n_labels = int(float(np.max(y)) + 1)
        if not os.path.exists(model_out_dir):
            os.mkdir(model_out_dir)
        return cls(vocab, c_args.tr_vec_file, c_args.val_vec_file,
                   coherence_via_encoder=c_args.encoder_coherence,
                   log_out_dir=log_out_dir,
                   model_out_dir=model_out_dir,
                   pretrained_param_file=c_args.pretrained_param_file, topic_seed_file=c_args.topic_seed_file,
                   use_labels_as_covars=c_args.use_labels_as_covars,
                   use_gpu=c_args.use_gpu, n_labels=n_labels, val_each_epoch=val_each_epoch)


    def pre_cache_vocabularies(self, sources):
        """Pre-cache pre-trained vocabularies.
        NOTE: This may cause problems with model selection when serializing a BowVAETrainer with
        large, memory-consuming pre-trained word embeddings.
        """
        for s in sources:
            self._initialize_vocabulary(s, set_vocab=False)

    def set_heldout_data_as_test(self):
        """Load in the heldout test data for final model evaluation
        """
        raise NotImplemented

    def _get_estimator(self, config, reporter, ctx):
        """Take a model configuration - specified by a config file or as determined by model selection and 
        return a VAE topic model ready for training.

        Parameters:
            config (dict): an autogluon configuration/argument object, instantiated to particular parameters
            reporter (`autogluon.core.scheduler.reporter.Reporter`): object for reporting model evaluations to scheduler
            ctx (`mxnet.context.Context`): Mxnet compute context
        
        Returns:
            Estimator (:class:`tmnt.estimator.BaseEstimator`): Either BowEstimator or CovariateBowEstimator
        """
        embedding_source = config['embedding.source']
        vocab, _ = self._initialize_vocabulary(embedding_source)
        if self.use_labels_as_covars:
            estimator = CovariateBowEstimator.from_config(self.n_labels, config, vocab,
                                                     pretrained_param_file=self.pretrained_param_file,
                                                     reporter=reporter, ctx=ctx)
        else:
           estimator = BowEstimator.from_config(config, vocab, n_labels = self.n_labels,
                                                coherence_via_encoder   = self.coherence_via_encoder,
                                                validate_each_epoch     = self.validate_each_epoch,
                                                pretrained_param_file   = self.pretrained_param_file,
                                                coherence_coefficient   = self.coherence_coefficient,
                                                reporter=reporter, ctx=ctx)
        return estimator
    

    def train_model(self, config, reporter):
        """Main training function which takes a single model configuration and a budget (i.e. number of epochs) and
        fits the model to the training data.
        
        Parameters:
            config: `Configuration` object within the specified `ConfigSpace`
            reporter: Reporter callback for model selection

        Returns:
            (tuple): Tuple containing:
                - model (:class:`tmnt.estimator.BowEstimator`) VAE model estimator with trained parameters
                - obj (float): scaled objective
                - results_details (dict): Dictionary of estimator metrics on validation data
        """
        logging.debug("Evaluating with Config: {}".format(config))
        ctx_list = self._get_mxnet_visible_gpus() if self.use_gpu else [mx.cpu()]
        ctx = ctx_list[0]
        vae_estimator = self._get_estimator(config, reporter, ctx)
        X, y = self._get_x_y_data(self.train_data_or_path)
        if self.test_data_or_path is None:
            vX, vy = None, None
        else:
            vX, vy = self._get_x_y_data(self.test_data_or_path)
        obj, v_res = vae_estimator.fit_with_validation(X, y, vX, vy)
        return vae_estimator, obj, v_res

    def write_model(self, estimator):
        """Method to write an estimated model to disk

        Parameters:
            estimator (tmnt.estimator.BowEstimator): BOW Estimator
        """
        if self.model_out_dir:
            mpath = Path(self.model_out_dir)
            mpath.mkdir(parents=True, exist_ok=True)
            estimator.write_model(self.model_out_dir)
        else:
            raise Exception("Model write failed, output directory not provided")



class SeqBowVEDTrainer(TopicTrainer):
    """Trainer for bag-of-words VAE topic models.
    
    Parameters:
        model_out_dir (str): Explicit string path to saved model information (and logging info).
        use_gpu (bool): Flag to force use of a GPU if available.  Default = False.
        log_interval (int): Perform validation (NPMI and perplexity) on the validation set this many batches. Default = 10.
        rng_seed (int): Seed for random number generator. Default = 1234
    """
    def __init__(self, model_out_dir, train_data_path, 
                 test_data_path, aux_data_path=None, use_gpu=False, log_interval=10, rng_seed=1234):
        super().__init__(None, model_out_dir, train_data_or_path=train_data_path, test_data_or_path=test_data_path,
                         aux_data_or_path=aux_data_path, use_gpu=use_gpu, val_each_epoch=True, rng_seed=rng_seed)
        self.model_out_dir = model_out_dir
        self.kld_wt = 1.0
        self.log_interval = log_interval


    @classmethod
    def from_arguments(cls, args, config):
        i_dt = datetime.datetime.now()
        train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,
                                                            i_dt.minute,i_dt.second)
        print("Set logging config to {}".format(train_out_dir))
        logging_config(folder=train_out_dir, name='train_trans_vae', level=args.log_level, console_level=args.log_level, no_console=False)
        logging.info(args)
        trainer = cls(
            train_out_dir,
            args.tr_file,
            args.val_file,
            aux_data_path = args.aux_file,
            use_gpu = args.use_gpu,
            log_interval = args.log_interval
            )
        return trainer


    def train_model(self, config, reporter):
        """Primary training call used for model training/evaluation by autogluon model selection
        or for training one off models.
        
        Parameters:
            config (:class:`tmnt.configuration.TMNTConfigSeqBOW`): TMNT configuration for seq-bow models
            reporter (:class:`autogluon.core.scheduler.reporter.Reporter`): object for reporting model evaluations to scheduler
        
        Returns:
            (tuple): Tuple containing:
                - model (:class:`tmnt.estimator.SeqBowEstimator`): variational BERT encoder-decoder model with trained parameters
                - obj (float): scaled objective with fit model
                - npmi (float): coherence on validation set
                - perplexity (float): perplexity score on validation data
                - redundancy (float): topic model redundancy of top 5 terms for each topic
        """
        ctx_list = self._get_mxnet_visible_gpus() if self.use_gpu else [mx.cpu()]
        ctx = ctx_list[0]
        vectorizer = TMNTVectorizer(vocab_size=4000, text_key="text", label_key="label")
        if isinstance(self.train_data_or_path, str):
            _, _ = vectorizer.fit_transform_json(self.train_data_or_path)
            classes = list(vectorizer.label_map) if config['use_labels'] else None
            tr_ds = JsonlDataset(self.train_data_or_path, txt_key="text", label_key="label")
            val_ds = JsonlDataset(self.test_data_or_path, txt_key="text", label_key="label")
            aux_ds = JsonlDataset(self.aux_data_or_path, txt_key="text", label_key="label") if self.aux_data_or_path else None
        else:
            train_data, train_labels = self.train_data_or_path
            val_data,  val_labels  = self.test_data_or_path
            tr_ds = ArrayDataset(train_data, train_labels)
            val_ds = ArrayDataset(val_data, val_labels)
            _,_ = vectorizer.fit_transform(train_data)            
            if self.aux_data_or_path:
                aux_data, _ = self.aux_data_or_path
                aux_ds, _ = vectorizer.transform(aux_data)
            else:
                aux_ds = None
            if train_labels:
                classes = list(set(train_labels))
                num_classes = len(classes)
            else:
                num_classes, classes = 0, None

        bert_model_name = config['bert_model_name']
        bert_dataset    = config['bert_dataset']
        batch_size      = config['batch_size']
        max_seq_len     = config['max_seq_len']
        
        tr_dataset, val_dataset, aux_dataset, num_examples, bert_base, bert_vocab  = \
            get_bert_datasets(classes, vectorizer, tr_ds, val_ds, batch_size, max_seq_len, aux_ds = aux_ds, 
                              bert_model_name=bert_model_name, bert_dataset=bert_dataset, ctx=ctx)
        n_labels = len(classes) if classes else 0
        logging.info('Number of labels: {}'.format(n_labels))
        logging.info('Number of examples: {}'.format(num_examples))
        seq_ved_estimator = SeqBowEstimator.from_config(config, bert_base, bert_vocab, vectorizer.get_vocab(), n_labels,
                                                        log_interval=self.log_interval,
                                                        reporter=reporter, ctx=ctx)
        obj, v_res = \
            seq_ved_estimator.fit_with_validation(tr_dataset, val_dataset, aux_dataset, num_examples)
        return seq_ved_estimator, obj, v_res


    def write_model(self, estimator, epoch_id=0):
        """Method to write an estimated model to disk along with configuration used to train the model and the vocabulary.

        Parameters:
            model (tmnt.modeling.BowVAEModel): Bag-of-words model (itself a gluon.block.Block)
            epoch_id (int): Id for printing out current epoch as checkpoint
        """
        model_dir = self.model_out_dir
        if model_dir:
            suf = '_'+ str(epoch_id) if epoch_id > 0 else ''
            estimator.write_model(model_dir, suffix=suf)


def train_bow_vae(args):
    try:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    except:
        logging.error("File passed to --config, {}, does not appear to be a valid .json configuration instance".format(args.config))
        raise Exception("Invalid Json Configuration File")
    dd = datetime.datetime.now()
    trainer = BowVAETrainer.from_arguments(args, val_each_epoch=args.eval_each_epoch)
    config = ag.space.Dict(**config_dict)
    estimator, obj, vres = trainer.train_with_single_config(config, args.num_final_evals)
    trainer.write_model(estimator)
    dd_finish = datetime.datetime.now()
    logging.info("Model training FINISHED. Time: {}".format(dd_finish - dd))
        

def train_seq_bow(c_args):
    try:
        with open(c_args.config, 'r') as f:
            config_dict = json.load(f)
    except:
        logging.error("File passed to --config, {}, does not appear to be a valid .json configuration instance"
                      .format(c_args.config))
        raise Exception("Invalid JSON configuration file")
    config = ag.space.Dict(**config_dict)    
    trainer = SeqBowVEDTrainer.from_arguments(c_args, config)
    estimator, obj, vres = trainer.train_with_single_config(config, 1)
    trainer.write_model(estimator)
    
