# coding: utf-8
# Copyright (c) 2020 The MITRE Corporation.
"""
Inferencers to make predictions and analyze data using trained topic models.
"""

import json
import mxnet as mx
import numpy as np
import gluonnlp as nlp
import io
import os
import scipy
import umap
import logging
import pickle
from tmnt.modeling import BowVAEModel, CovariateBowVAEModel, SeqBowVED, MetricSeqBowVED
from tmnt.data_loading import DataIterLoader, file_to_data, SparseMatrixDataIter
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.distribution import HyperSphericalDistribution
from tmnt.utils.recalibrate import recalibrate_scores
from multiprocessing import Pool
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from sklearn.datasets import load_svmlight_file


MAX_DESIGN_MATRIX = 250000000 

class BaseInferencer(object):
    """Base inference object for text encoding with a trained topic model.

    """
    def __init__(self, ctx):
        self.ctx = ctx

    def encode_texts(self, intexts):
        raise NotImplementedError

    def get_top_k_words_per_topic(self, k):
        raise NotImplementedError

    def get_top_k_words_per_topic_per_covariate(self, k):
        raise NotImplementedError


class BowVAEInferencer(BaseInferencer):
    """
    """
    def __init__(self, model, pre_vectorizer=None):
        super().__init__(model.model_ctx)
        self.max_batch_size = 16
        self.vocab = model.vocabulary
        self.vectorizer = pre_vectorizer or TMNTVectorizer(initial_vocabulary=model.vocabulary)
        self.n_latent = model.n_latent
        self.model = model
        if isinstance(model, CovariateBowVAEModel):
            self.covar_model = True
            self.n_covars = model.n_covars
            self.covar_net_layers = model.covar_net_layers
        else:
            self.covar_model = False
        

    @classmethod
    def from_saved(cls, param_file=None, config_file=None, vocab_file=None, model_dir=None, ctx=mx.cpu()):
        if model_dir is not None:
            param_file = os.path.join(model_dir,'model.params')
            vocab_file = os.path.join(model_dir,'vocab.json')
            config_file = os.path.join(model_dir,'model.config')
            serialized_vectorizer_file = os.path.join(model_dir,'vectorizer.pkl')
        with open(config_file) as f:
            config = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        vocab = nlp.Vocab.from_json(voc_js)
        n_latent = config['n_latent']
        enc_dim = config['enc_hidden_dim']
        lat_distrib = config['latent_distribution']['dist_type']
        n_encoding_layers = config.get('num_enc_layers', 0)
        enc_dr= float(config.get('enc_dr', 0.0))
        emb_size = config['derived_info']['embedding_size']
        n_labels = config.get('n_labels') or 0
        gamma    = config.get('gamma') or 1.0
        multilabel = config.get('multilabel') or False
        if 'n_covars' in config:
            n_covars = config['n_covars']
            covar_net_layers = config.get('covar_net_layers')
            model = CovariateBowVAEModel(covar_net_layers, 
                                         enc_dim, emb_size, n_encoding_layers, enc_dr, False,
                                         vocabulary=vocab, n_covars=n_covars, latent_distrib=lat_distrib,
                                         n_latent=n_latent,
                                         ctx=ctx)
        else:
            model = BowVAEModel(enc_dim, emb_size, n_encoding_layers, enc_dr, False, n_labels=n_labels, gamma=gamma,
                                multilabel=multilabel,
                                vocabulary=vocab, latent_distrib=lat_distrib, n_latent=n_latent,
                                ctx=ctx)
        model.load_parameters(str(param_file), allow_missing=False)
        return cls(model, pre_vectorizer=vectorizer)


    def get_model_details(self, sp_vec_file):
        data_csr, labels = load_svmlight_file(sp_vec_file, n_features=len(self.vocab))
        data_csr = mx.nd.sparse.csr_matrix(data_csr, dtype='float32')
        ## 1) K x W matrix of P(term|topic) probabilities
        w = self.model.decoder.collect_params().get('weight').data().transpose() ## (K x W)
        w_pr = mx.nd.softmax(w, axis=1)
        ## 2) D x K matrix over the test data of topic probabilities
        covars = labels if self.covar_model else None
        dt_matrix = self.encode_data(data_csr, covars, use_probs=True)
        ## 3) D-length vector of document sizes
        doc_lengths = data_csr.sum(axis=1)
        ## 4) vocab (in same order as W columns)
        ## 5) frequency of each word w_i \in W over the test corpus
        term_cnts = data_csr.sum(axis=0)
        return w_pr, dt_matrix, doc_lengths, term_cnts


    def get_pyldavis_details(self, sp_vec_file):
        w_pr, dt_matrix, doc_lengths, term_cnts = self.get_model_details(sp_vec_file)
        d1 = w_pr.asnumpy().tolist()
        d2 = list(map(lambda x: x.asnumpy().tolist(), dt_matrix))
        d3 = doc_lengths.asnumpy().tolist()
        d5 = term_cnts.asnumpy().tolist()
        d4 = list(map(lambda i: self.vocab.idx_to_token[i], range(len(self.vocab.idx_to_token))))
        d = {'topic_term_dists': d1, 'doc_topic_dists': d2, 'doc_lengths': d3, 'vocab': d4, 'term_frequency': d5 }
        return d


    def get_umap_embeddings(self, data, umap_metric='euclidean'):
        encs = self.encode_data(data, None)
        encs2 = np.array([enc.asnumpy() for enc in encs])
        um = umap.UMAP(n_neighbors=4, min_dist=0.1, metric='euclidean')
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

    def encode_texts(self, texts, use_probs=False, include_bn=False):
        X, _ = self.vectorizer.transform(texts)
        encodings = self.encode_data(X, None, use_probs=use_probs, include_bn=include_bn)
        return encodings

    def _get_data_iterator(self, data_mat, labels):
        x_size = data_mat.shape[0] * data_mat.shape[1]
        if x_size <= MAX_DESIGN_MATRIX and isinstance(data_mat, scipy.sparse.csr.csr_matrix):
            data_mat = mx.nd.sparse.csr_matrix(data_mat, dtype='float32')
        elif isinstance(data_mat, mx.nd.NDArray):
            data_mat = mx.nd.array(data_mat, dtype='float32')
        batch_size = min(data_mat.shape[0], self.max_batch_size)
        last_batch_size = data_mat.shape[0] % batch_size
        covars = mx.nd.one_hot(mx.nd.array(labels, dtype='int'), self.n_covars) \
            if self.covar_model and labels[:-last_batch_size] is not None else None
        if last_batch_size < 1: 
            data_to_iter = data_mat 
        else:
            data_to_iter = data_mat[:-last_batch_size]
        if x_size > MAX_DESIGN_MATRIX:
            logging.info("Sparse matrix has total size = {}. Using Sparse Matrix data batcher.".format(x_size))
            if covars is None:
                covars = mx.nd.zeros(data_to_iter.shape[0])
            infer_iter = DataIterLoader(SparseMatrixDataIter(data_to_iter, covars, batch_size, last_batch_handle='discard',
                                                             shuffle=False))
        else:
            infer_iter = DataIterLoader(mx.io.NDArrayIter(data_to_iter, covars,
                                                      batch_size, last_batch_handle='discard', shuffle=False))
        return infer_iter, last_batch_size

    def encode_data(self, data_mat, labels, use_probs=True, include_bn=False):
        infer_iter, last_batch_size = self._get_data_iterator(data_mat, labels)
        encodings = []
        for _, (data,labels) in enumerate(infer_iter):
            data = data.as_in_context(self.ctx)
            if self.covar_model and labels is not None:
                labels = labels.as_in_context(self.ctx)
                encs = self.model.encode_data_with_covariates(data, labels, include_bn=include_bn)
            else:
                encs = self.model.encode_data(data, include_bn=include_bn)
            if use_probs:
                e1 = (encs - mx.nd.min(encs, axis=1).expand_dims(1)).astype('float64')
                encs = list(mx.nd.softmax(e1).asnumpy())
                encs = list(map(recalibrate_scores, encs))
            else:
                encs = list(encs.astype('float64').asnumpy())
            encodings.extend(encs)
        ## handle the last batch explicitly as NDArrayIter doesn't do that for us
        if last_batch_size > 0:
            last_data = mx.nd.sparse.csr_matrix(data_mat[-last_batch_size:], dtype='float32')
            data = last_data.as_in_context(self.ctx)
            if self.covar_model and labels is not None:
                labels = mx.nd.one_hot(mx.nd.array(labels[-last_batch_size:], dtype='int'),
                                       self.n_covars).as_in_context(self.ctx)
                encs = self.model.encode_data_with_covariates(data, labels)
            else:
                encs = self.model.encode_data(data)
            if use_probs:
                e1 = (encs - mx.nd.min(encs, axis=1).expand_dims(1)).astype('float64')
                encs = list(mx.nd.softmax(e1).asnumpy())
                encs = list(map(recalibrate_scores, encs))
            else:
                encs = list(encs.astype('float64').asnumpy())
            encodings.extend(encs)
        return encodings

    def get_likelihood_stats(self, data_mat, n_samples=50):
        ## Notes:
        ## Following ideas in the paper:
        ## Bayesian Autoencoders: Analysing and Fixing the Bernoulli likelihood for Out-of-Distribution Detection
        ## But - that analysis was done on images with less sparsity
        ## Consider using Gaussian liklihood here as well to avoid skewness associated with Bernoulli likilhood
        data_iter, last_batch_size = self._get_data_iterator(data_mat, None)
        all_stats = []
        for _, (data, labels) in enumerate(data_iter):
            elbos = []
            for s in range(0,n_samples):
                elbo, _,_,_,_,_,_ = self.model(data.as_in_context(self.ctx), labels)
                elbos.append(list(elbo.asnumpy()))
            wd_cnts = data.sum(axis=1).asnumpy()
            elbos_np = np.array(elbos) / (wd_cnts + 1)
            elbos_means = list(elbos_np.mean(axis=0))
            elbos_var   = list(elbos_np.var(axis=0))
            all_stats.extend(list(zip(elbos_means, elbos_var)))
        return all_stats


    def get_top_k_words_per_topic(self, k):
        sorted_ids = self.model.get_ordered_terms()
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t]) ]
            topic_terms.append(top_k)
        return topic_terms

    def get_top_k_words_per_topic_encoder(self, k, dataloader, sample_size=-1):
        sorted_ids = self.model.get_ordered_terms_encoder(dataloader, sample_size=sample_size)
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t]) ]
            topic_terms.append(top_k)
        return topic_terms


    def get_top_k_words_per_topic_per_covariate(self, k):
        n_topics = self.n_latent
        w = self.model.cov_decoder.cov_inter_decoder.collect_params().get('weight').data()
        n_covars = int(w.shape[1] / n_topics)
        topic_terms = []
        for i in range(n_covars):
            cv_i_slice = w[:, (i * n_topics):((i+1) * n_topics)]
            sorted_ids = cv_i_slice.argsort(axis=0, is_ascend=False)
            cv_i_terms = []
            for t in range(n_topics):
                top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
                cv_i_terms.append(top_k)
            topic_terms.append(cv_i_terms)
        return topic_terms

    def get_top_k_words_per_topic_over_scalar_covariate(self, k, min_v=0.0, max_v=1.0, step=0.1):
        raise NotImplemented

    def predict_text(self, txt, pred_threshold=0.5):
        X_csr, _      = self.vectorizer.transform(txt)
        X = mx.nd.sparse.csr_matrix(X_csr, dtype='float32')
        encodings = self.encode_data(X, None, use_probs=True, include_bn=False)
        preds     = self.model.predict(X).asnumpy()
        inv_map = [0] * len(self.vectorizer.label_map)
        for k in self.vectorizer.label_map:
            inv_map[self.vectorizer.label_map[k]] = k
        if not self.model.multilabel:
            best = np.argmax(preds)
            best_strs = [ inv_map[best] ]
        else:
            best_strs = [ inv_map[i] for i in list(np.where(preds > pred_threshold)[0]) ]
        return best_strs, encodings
    


class SeqVEDInferencer(BaseInferencer):
    """Inferencer for sequence variational encoder-decoder models using BERT
    """
    def __init__(self, model, bert_vocab, max_length, bow_vocab=None, pre_vectorizer=None, ctx=mx.cpu()):
        super().__init__(ctx)
        self.model     = model
        self.bert_base = model.bert
        self.tokenizer = BERTTokenizer(bert_vocab)
        self.transform = BERTSentenceTransform(self.tokenizer, max_length, pair=False)
        self.bow_vocab = bow_vocab
        self.vectorizer = pre_vectorizer or TMNTVectorizer(initial_vocabulary=bow_vocab)


    @classmethod
    def from_saved(cls, param_file=None, config_file=None, vocab_file=None, model_dir=None, max_length=128, ctx=mx.cpu()):
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
        bert_base, vocab = nlp.model.get_model(config['bert_model_name'],  
                                               dataset_name=config['bert_data_name'],
                                               pretrained=True, ctx=ctx, use_pooler=True,
                                               use_decoder=False, use_classifier=False)
        latent_dist_t = config['latent_distribution']['dist_type']       
        n_latent    = config['n_latent']
        kappa       = config['latent_distribution']['kappa']
        num_classes = config['n_labels']
        classifier_dropout = config['classifier_dropout']
        pad_id      = vocab[vocab.padding_token]
        latent_dist = HyperSphericalDistribution(n_latent, kappa=kappa, ctx=ctx)
        model = SeqBowVED(bert_base, latent_dist=latent_dist, bow_vocab_size = len(bow_vocab), num_classes=num_classes,
                          dropout=classifier_dropout)
        model.load_parameters(str(param_file), allow_missing=False, ignore_extra=True)
        return cls(model, vocab, max_length, bow_vocab, pre_vectorizer=vectorizer, ctx=ctx)


    def _embed_sequence(self, ids, segs):
        embeddings = self.bert_base.word_embed(ids) + self.bert_base.token_type_embed(segs)
        return embeddings.transpose((1,0,2))

    def _encode_from_embedding(self, embeddings, lens):
        outputs, _ = self.bert_base.encoder(embeddings, valid_length=lens)
        outputs = outputs.transpose((1,0,2))
        # outputs should be (batch, seq_len, C) shaped now
        pooled_out = self.bert_base._apply_pooling(outputs)
        topic_encoding = self.model.latent_dist.get_mu_encoding(pooled_out)
        return topic_encoding


    def prep_text(self, txt):    # used for integrated gradients
        tokens = self.tokenizer(txt)
        ids, lens, segs = self.transform((txt,))
        return ( tokens,
                 mx.nd.array([ids], dtype='int32'),
                 mx.nd.array([lens], dtype='float32'),
                 mx.nd.array([segs], dtype='int32') )
    

    def encode_text(self, txt):                   
        tokens, ids, lens, segs = self.prep_text(txt)
        _, enc = self.model.bert(ids.as_in_context(self.ctx),
                                              segs.as_in_context(self.ctx), lens.as_in_context(self.ctx))
        topic_encoding = self.model.latent_dist.get_mu_encoding(enc)
        return topic_encoding, tokens

    def predict_text(self, txt):
        encoding, _ = self.encode_text(txt)
        return self.model.classifier(encoding)

    def get_likelihood_stats(self, txt, n_samples=50):
        tokens, ids, lens, segs = self.prep_text(txt)
        bow_vector = self.vectorizer.vectorizer.transform([txt])
        elbos = []
        for s in range(n_samples):
            elbo, _, _, _, _ = self.model(tokens, segs, lens, bow_vector)
            elbos.append(list(elbo.asnumpy()))
        wd_cnts = bow.sum(axis=1).asnumpy()
        elbos_np = np.array(elbos) / (wd_cnts + 1)
        elbos_means = list(elbos_np.mean(axis=0))
        elbos_var   = list(elbos_np.var(axis=0))
        return elbos_means, elbos_var
        

    def encode_data(self, dataloader, use_probs=False):
        encodings = []
        for _, seqs in enumerate(dataloader):
            ids, lens, segs, _, _ = seqs
            _, encs = self.model.bert(ids.as_in_context(self.ctx),
                                      segs.as_in_context(self.ctx), lens.astype('float32').as_in_context(self.ctx))
            raw_topic_encodings = list(self.model.latent_dist.get_mu_encoding(encs))
            renormed_topic_encodings = map(recalibrate, raw_topic_encodings)
            encodings.extend(renormed_topic_encod)
        return encodings

    def get_top_k_words_per_topic(self, k):
        if self.bow_vocab:
            sorted_ids = self.model.get_top_k_terms(k)
            topic_terms = []
            for t in range(self.model.n_latent):
                top_k = [ self.bow_vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t]) ]
                topic_terms.append(top_k)
            return topic_terms
        else:
            raise Exception("Bow vocabulary required for Inferencer in order to provide topic terms")


class MetricSeqVEDInferencer(SeqVEDInferencer):
    """Inferencer for sequence variational encoder-decoder models using BERT trained via Metric Learning
    """
    def __init__(self, model, bert_vocab, max_length, bow_vocab=None, ctx=mx.cpu()):
        super().__init__(model, bert_vocab, max_length, bow_vocab=bow_vocab, ctx=ctx)

    @classmethod
    def from_saved(cls, param_file=None, config_file=None, vocab_file=None, model_dir=None, max_length=128, ctx=mx.cpu()):
        if model_dir is not None:
            param_file = os.path.join(model_dir, 'model.params')
            vocab_file = os.path.join(model_dir, 'vocab.json')
            config_file = os.path.join(model_dir, 'model.config')
        with open(config_file) as f:
            config = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        bow_vocab = nlp.Vocab.from_json(voc_js)
        bert_base, vocab = nlp.model.get_model(config['bert_model_name'],  
                                               dataset_name=config['bert_data_name'],
                                               pretrained=True, ctx=ctx, use_pooler=True,
                                               use_decoder=False, use_classifier=False) #, output_attention=True)
        latent_dist_t = config['latent_distribution']['dist_type']       
        n_latent    = config['n_latent']
        kappa       = config['latent_distribution']['kappa']
        num_classes = config['n_labels']
        classifier_dropout = config['classifier_dropout']
        pad_id      = vocab[vocab.padding_token]
        latent_dist = HyperSphericalDistribution(n_latent, kappa=kappa, ctx=ctx)
        model = MetricSeqBowVED(bert_base, latent_dist=latent_dist, bow_vocab_size = len(bow_vocab), n_latent=n_latent,
                                dropout=classifier_dropout)
        model.load_parameters(str(param_file), allow_missing=False, ignore_extra=True)
        return cls(model, vocab, max_length, bow_vocab, ctx)


        

    
