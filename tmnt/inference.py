# coding: utf-8
# Copyright (c) 2020-2021 The MITRE Corporation.
"""
Inferencers to make predictions and analyze data using trained topic models.
"""

import json
import numpy as np
import io
import os
import torch
import pickle
from tmnt.modeling import BowVAEModel, SeqBowVED, MetricSeqBowVED
from tmnt.estimator import BowEstimator, SeqBowEstimator, SeqBowMetricEstimator
from tmnt.data_loading import SparseDataLoader
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.utils.recalibrate import recalibrate_scores
from sklearn.datasets import load_svmlight_file
from functools import partial
from tmnt.data_loading import get_llm_tokenizer
from typing import List, Tuple, Dict, Optional, Union, NoReturn
from scipy.sparse import csr_matrix
from tmnt.distribution import ConceptLogisticGaussianDistribution


MAX_DESIGN_MATRIX = 250000000 

class BaseInferencer(object):
    """Base inference object for text encoding with a trained topic model.

    """
    def __init__(self, estimator, vectorizer, device):
        self.device = device
        self.estimator = estimator
        self.vectorizer = vectorizer
        

    def save(self, model_dir: str) -> None:
        """
        Save model and vectorizer to disk

        Parameters:
            model_dir: Model directory to save parmaeters, config, vocabulary and vectorizer
        """
        self.estimator.write_model(model_dir)
        serialized_vector_file = os.path.join(model_dir, 'vectorizer.pkl')
        if self.vectorizer:
            with io.open(serialized_vector_file, 'wb') as fp:
                pickle.dump(self.vectorizer, fp)


    def encode_texts(self, intexts):
        raise NotImplementedError

    def get_top_k_words_per_topic(self, k):
        raise NotImplementedError

    def get_pyldavis_details(self, sp_vec_file_or_X, y=None):
        w_pr, dt_matrix, doc_lengths, term_cnts = self.get_model_details(sp_vec_file_or_X, y=y)
        d1 = w_pr.cpu().detach().numpy().tolist()
        d2 = list(map(lambda x: x.tolist(), dt_matrix))
        doc_lengths = np.array(doc_lengths)
        d3 = list(doc_lengths.squeeze())
        d5 = term_cnts.squeeze().tolist()
        d4 = list(map(lambda i: self.vocab.lookup_token(i), range(len(self.vocab))))
        d = {'topic_term_dists': d1, 'doc_topic_dists': d2, 'doc_lengths': d3, 'vocab': d4, 'term_frequency': d5 }
        return d


class BowVAEInferencer(BaseInferencer):
    """
    """
    def __init__(self, estimator, pre_vectorizer=None):
        super().__init__(estimator,
                         pre_vectorizer or TMNTVectorizer(initial_vocabulary=estimator.model.vocabulary),
                         estimator.model.device)
        self.max_batch_size = 16
        self.vocab = estimator.vocabulary
        self.n_latent = estimator.n_latent
        self.model = estimator.model
        self.covar_model = False

    @classmethod
    def from_saved(cls, model_dir=None, device='cpu'):
        serialized_vectorizer_file = None
        config_file = os.path.join(model_dir,'model.config')
        param_file = os.path.join(model_dir,'model.params')
        vocab_file = os.path.join(model_dir,'vocab.json')
        serialized_vectorizer_file = os.path.join(model_dir,'vectorizer.pkl')
        with io.open(config_file, 'r') as f:
            config_dict = json.load(f)
        estimator = BowEstimator.from_saved(model_dir)
        estimator.initialize_with_pretrained()
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        return cls(estimator, pre_vectorizer=vectorizer)


    def get_model_details(self, sp_vec_file_or_X, y=None):
        data_csr, labels = sp_vec_file_or_X, y
        #data_csr = mx.nd.sparse.csr_matrix(data_csr, dtype='float32')
        ## 1) K x W matrix of P(term|topic) probabilities
        w = self.model.decoder.weight.data.t() ## (K x W)
        w_pr = torch.softmax(w, dim=1)
        ## 2) D x K matrix over the test data of topic probabilities
        covars = labels if self.covar_model else None
        #dt_matrix, _ = self.encode_data(data_csr, covars, use_probs=True, target_entropy=2.0)
        dt_matrix, _ = self.encode_data(data_csr, covars, use_probs=True, target_entropy=2.5)
        ## 3) D-length vector of document sizes
        doc_lengths = data_csr.sum(axis=1)
        ## 4) vocab (in same order as W columns)
        ## 5) frequency of each word w_i \in W over the test corpus
        term_cnts = np.array(data_csr.sum(axis=0))
        return w_pr, dt_matrix, doc_lengths, term_cnts

    def get_umap_embeddings(self, data, umap_metric='euclidean', use_probs=False, target_entropy=1.0):
        encs = self.encode_data(data, None, use_probs=use_probs, target_entropy=target_entropy)
        encs2 = np.array(encs)
        um = umap.UMAP(n_neighbors=4, min_dist=0.2, metric='euclidean')
        return um.fit_transform(encs2)

    def plot_to(self, embeddings, labels, f=None):
        import matplotlib.pyplot as plt
        plt.scatter(*embeddings.T, c=labels, s=0.8, alpha=0.9, cmap='coolwarm')
        if f is None:
            plt.show()
        else:
            plt.savefig(f)

    def export_full_model_inference_details(self, sp_vec_file, ofile):
        d = self.get_pyldavis_details(sp_vec_file)
        with io.open(ofile, 'w') as fp:
            json.dump(d, fp, sort_keys=True, indent=4)        

    def encode_vec_file(self, sp_vec_file, use_probs=False):
        data_mat, labels = load_svmlight_file(sp_vec_file, n_features=len(self.vocab), zero_based=True)
        return self.encode_data(data_mat, labels, use_probs=use_probs), labels

    def encode_texts(self, texts, use_probs=True, include_bn=True, target_entropy=1.5):
        X, _ = self.vectorizer.transform(texts)
        encodings = self.encode_data(X, None, use_probs=use_probs, include_bn=include_bn, 
                                     target_entropy=target_entropy)
        return encodings

    def _get_data_iterator(self, data_mat, labels):
        x_size = data_mat.shape[0] * data_mat.shape[1]
        batch_size = min(data_mat.shape[0], self.max_batch_size)
        last_batch_size = data_mat.shape[0] % batch_size
        covars = None
        #torch.one_hot(torch.Tensor(labels, dtype='int'), num_classes=self.n_covars) \
        #    if self.covar_model is not None else None
        infer_iter = SparseDataLoader(data_mat, covars, batch_size=batch_size)
        return infer_iter, last_batch_size

    def encode_data(self, data_mat, labels=None, use_probs=True, include_bn=True, target_entropy=1.0, include_predictions=False):
        infer_iter, last_batch_size = self._get_data_iterator(data_mat, labels)
        encodings = []
        predictions = []
        for _, (data,labels) in enumerate(infer_iter):
            with torch.no_grad():
                data = data.to(self.device)
                encs = self.model.encode_data(data, include_bn=include_bn)
                if include_predictions:
                    if self.model.multilabel:
                        preds = list(self.model.predict(data).sigmoid().detach().numpy())
                    else:
                        preds = list(self.model.predict(data).softmax(dim=1).detach().numpy())
                if use_probs:
                    #e1 = (encs - encs.min(dim=1).unsqueeze(1)).astype('float64')
                    e1 = (encs - encs.min(dim=1)[0].unsqueeze(1))
                    encs = list(torch.nn.functional.softmax(e1, dim=-1).numpy())
                    encs = list(map(partial(recalibrate_scores, target_entropy=target_entropy), encs))
                else:
                    encs = list(encs.numpy())
                encodings.extend(encs)
                if include_predictions:
                    predictions.extend(preds)
        return encodings, predictions
    

    def get_likelihood_stats(self, data_mat, n_samples=50):
        """Get the expected elbo and its variance for input data using sampling
        Parameters:
            data_mat: Document term matrix 
        """
        data_iter, last_batch_size = self._get_data_iterator(data_mat, None)
        all_stats = []
        for _, (data, labels) in enumerate(data_iter):
            elbos = []
            for s in range(0,n_samples):
                elbo, _,_,_,_,_,_ = self.model(data.as_in_context(self.device), labels)
                elbos.append(list(elbo.asnumpy()))
            wd_cnts = data.sum(dim=1).asnumpy()
            elbos_np = np.array(elbos) / (wd_cnts + 1)
            elbos_means = list(elbos_np.mean(dim=0))
            elbos_var   = list(elbos_np.var(dim=0))
            all_stats.extend(list(zip(elbos_means, elbos_var)))
        return all_stats

    def get_top_k_words_per_topic(self, k):
        sorted_ids = self.model.get_ordered_terms()
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.lookup_token(int(i)) for i in list(sorted_ids[:k, t]) ]
            topic_terms.append(top_k)
        return topic_terms

    def get_top_k_words_per_topic_encoder(self, k, dataloader, sample_size=-1):
        sorted_ids = self.model.get_ordered_terms_encoder(dataloader, sample_size=sample_size)
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.lookup_token(int(i)) for i in list(sorted_ids[:k, t]) ]
            topic_terms.append(top_k)
        return topic_terms


    def predict_text(self, txt: List[str], pred_threshold: float = 0.5) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
        """Take a list of input documents/passages as strings and return document encodings (topics) and classification outputs
        
        Parameters:
            txt: List of input document strings
            pred_threshold: Threshold to use for multilabel classification
        Returns:
            top predicted labels, encodings, posteriors
        """
        X, _      = self.vectorizer.transform(txt)
        #X = mx.nd.sparse.csr_matrix(X_csr, dtype='float32')
        encodings, preds = self.encode_data(X, None, use_probs=True, include_bn=True, include_predictions=True)
        inv_map = [0] * len(self.vectorizer.label_map)
        for k in self.vectorizer.label_map:
            inv_map[self.vectorizer.label_map[k]] = k
        if not self.model.multilabel:
            bests = np.argmax(preds, axis=1)
            best_strs = [ inv_map[int(best)] for best in bests ]
        else:
            best_strs = [ inv_map[i] for i in list(np.where(np.array(preds) > pred_threshold)[0]) ]
        return best_strs, encodings, preds
    


class SeqVEDInferencer(BaseInferencer):
    """Inferencer for sequence variational encoder-decoder models using a pretrained Transformer model
    """
    def __init__(self, estimator, max_length, pre_vectorizer=None, device='cpu'):
        super().__init__(estimator,
                         pre_vectorizer or TMNTVectorizer(initial_vocabulary=estimator.vocabulary),
                         device)
        self.model     = estimator.model 
        self.llm = self.model.llm
        self.vocab = estimator.vocabulary
        self.tokenizer = get_llm_tokenizer(estimator.llm_model_name)
        self.txt_max_len = max_length


    @classmethod
    def from_saved(cls, model_dir=None, max_length=128, device='cpu'):
        estimator = SeqBowEstimator.from_saved(model_dir=model_dir, device=device)
        serialized_vectorizer_file = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        return cls(estimator, max_length, pre_vectorizer=vectorizer, device=device)


    def prep_text(self, txt): 
        tokenized_result = self.tokenizer(txt,return_tensors='pt', padding='max_length',
                                           max_length=self.txt_max_len, truncation=True)
        return tokenized_result

    def encode_text(self, txt, as_numpy=False):                   
        token_result = self.prep_text(txt)
        self.model.eval()
        topic_encoding = self.model.forward_encode(token_result['input_ids'].to(self.device), 
                                                   token_result['attention_mask'].to(self.device))
        return topic_encoding.cpu().detach().numpy() if as_numpy else topic_encoding

    def predict_text(self, txt):
        encoding = self.encode_text(txt)
        return self.model.classifier(encoding)

    def encode_data(self, dataloader, use_probs=False, target_entropy=2.0):
        encodings = []
        bow_matrix = []
        for _, data_batch in enumerate(dataloader):
            seqs, = data_batch
            _, input_ids, mask, bow = seqs
            encs = self.model.forward_encode(input_ids.to(self.device), mask.to(self.device))
            if use_probs:
                e1 = (encs - encs.min(dim=1)[0].unsqueeze(1))
                encs = list(torch.nn.functional.softmax(e1, dim=-1).cpu().detach().numpy())
                encs = list(map(partial(recalibrate_scores, target_entropy=target_entropy), encs))
            else:
                encs = encs.cpu().detach().numpy()
            bow_matrix.append(bow.to_dense().cpu().detach().numpy().squeeze())
            encodings.extend(encs)
        return np.vstack(encodings), np.vstack(bow_matrix)

    def get_model_details(self, dataloader, y=None):
        #data_csr = mx.nd.sparse.csr_matrix(data_csr, dtype='float32')
        ## 1) K x W matrix of P(term|topic) probabilities
        w = self.model.decoder.weight.data.t() ## (K x W)
        w_pr = torch.softmax(w, dim=1)
        ## 2) D x K matrix over the test data of topic probabilities
        #dt_matrix, _ = self.encode_data(data_csr, covars, use_probs=True, target_entropy=2.0)
        dt_matrix, bow_matrix = self.encode_data(dataloader, use_probs=True, target_entropy=2.5)
        ## 3) D-length vector of document sizes
        doc_lengths = bow_matrix.sum(axis=1)
        ## 4) vocab (in same order as W columns)
        ## 5) frequency of each word w_i \in W over the test corpus
        term_cnts = np.array(bow_matrix.sum(axis=0))
        return w_pr, dt_matrix, doc_lengths, term_cnts

    def get_top_k_words_per_topic(self, k):
        if self.vocab:
            sorted_ids = self.model.get_ordered_terms()
            topic_terms = []
            for t in range(self.model.n_latent):
                top_k = self.vocab.lookup_tokens(sorted_ids[:k, t]) 
                topic_terms.append(top_k)
            return topic_terms
        else:
            raise Exception("Bow vocabulary required for Inferencer in order to provide topic terms")


class MetricSeqVEDInferencer(SeqVEDInferencer):
    """Inferencer for sequence variational encoder-decoder models using BERT trained via Metric Learning
    """
    def __init__(self, estimator, max_length, pre_vectorizer=None, device='cpu'):
        super().__init__(estimator, max_length, pre_vectorizer=pre_vectorizer, device=device)


    @classmethod
    def from_saved(cls, param_file=None, config_file=None, vocab_file=None, model_dir=None, max_length=128, device='cpu'):
        estimator = SeqBowMetricEstimator.from_saved(model_dir=model_dir, device=device)
        serialized_vectorizer_file = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        return cls(estimator, max_length, pre_vectorizer=vectorizer, device=device)







        

    
