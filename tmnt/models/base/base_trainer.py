# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import logging
import mxnet as mx
import datetime
import time
import copy
from tmnt.utils.random import seed_rng
from autogluon.scheduler.reporter import FakeReporter
import gluonnlp as nlp

class BaseTrainer(object):
    """Abstract base class for topic model trainers. 

    Objects of this class provide all functionality for training a topic model, including 
    handle data prep/loading, model definition, and training parameters.

    Args:
        vocabulary (`gluonnlp.Vocab`): Gluon NLP vocabulary object representing the bag-of-words used for the dataset
        train_data (array-like or sparse matrix): Training input data tensor
        test_data (array-like or sparse matrix): Testing/validation input data tensor
    """

    def __init__(self, vocabulary, train_data, test_data, train_labels, test_labels, rng_seed):
        self.train_data   = train_data
        self.test_data    = test_data
        self.train_labels = train_labels
        self.test_labels  = test_labels
        self.rng_seed     = rng_seed
        self.vocabulary   = vocabulary
        self.vocab_cache  = {}


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
        

    def x_get_mxnet_visible_gpus(self):
        """Gets visible gpus from MXNet.
        
        Returns:
            `mxnet.context.Context` representing the GPU context
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
        """Fit models with a single configuration and report the value of the objective function.

        This method trains a model defined by the configuration `num_evals` times. Each time
        the model weights are randomly initialized with a different RNG seed. The results
        of each run are captured and mean and std reported.
        
        Args:
            config (dict): Configuration instance with hyperparameter values for model definition.
            num_evals (int): Number of model fits and evaluations to perform (with random initialization)

        Returns:
            (tuple): Tuple containing:
                model (:class:`tmnt.models.base.base_vae.BaseVAE`): VAE Model instance with trained/fit parameters.
                objective (float): Value of the objective function with the best model.
        """
        rng_seed = self.rng_seed
        best_obj = -1000000000.0
        best_model = None
        if self.test_data is not None:
            #if c_args.tst_vec_file:
            #    trainer.set_heldout_data_as_test()        
            logging.info("Training with config: {}".format(config))
            npmis, perplexities, redundancies, objectives = [],[],[],[]
            ntimes = int(num_evals)
            for i in range(ntimes):
                seed_rng(rng_seed) # update RNG
                rng_seed += 1
                model, obj, npmi, perplexity, redundancy = self.train_model(config, FakeReporter())
                npmis.append(npmi)
                perplexities.append(perplexity)
                redundancies.append(redundancy)
                objectives.append(obj)
                if obj > best_obj:
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
                logging.info("Final {} NPMI         ==> {}".format(test_type, npmis[0]))
                logging.info("Final {} Perplexity   ==> {}".format(test_type, perplexities[0]))
                logging.info("Final {} Redundancy   ==> {}".format(test_type, redundancies[0]))
                logging.info("Final {} Objective    ==> {}".format(test_type, objectives[0]))            
            return best_model, best_obj
        else:
            model, obj, _, _, _ = self.train_model(config, FakeReporter())
            return model, obj

        

    def train_model(self, config, reporter):
        """
        Parameters
        ----------
        
        """
        raise NotImplementedError()
