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
from tmnt.modeling import BowVAEModel, CovariateBowVAEModel, BertBowVED
from tmnt.data_loading import DataIterLoader, file_to_data
from tmnt.preprocess.vectorizer import TMNTVectorizer
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
    def __init__(self, model, ctx=mx.cpu()):
        super().__init__(ctx)
        self.max_batch_size = 16
        self.vocab = model.vocabulary
        self.vectorizer = TMNTVectorizer(initial_vocabulary=model.vocabulary)
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
        with open(config_file) as f:
            config = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        vocab = nlp.Vocab.from_json(voc_js)
        n_latent = config['n_latent']
        enc_dim = config['enc_hidden_dim']
        lat_distrib = config['latent_distribution']['dist_type']
        n_encoding_layers = config.get('num_enc_layers', 0)
        enc_dr= float(config.get('enc_dr', 0.0))
        emb_size = config['derived_info']['embedding_size']
        if 'n_covars' in config:
            n_covars = config['n_covars']
            covar_net_layers = config.get('covar_net_layers')
            model = CovariateBowVAEModel(covar_net_layers, 
                                         enc_dim, emb_size, n_encoding_layers, enc_dr, False,
                                         vocabulary=vocab, n_covars=n_covars, latent_distrib=lat_distrib,
                                         n_latent=n_latent,
                                         ctx=ctx)
        else:
            model = BowVAEModel(enc_dim, emb_size, n_encoding_layers, enc_dr, False,
                                vocabulary=vocab, latent_distrib=lat_distrib, n_latent=n_latent,
                                ctx=ctx)
        model.load_parameters(str(param_file), allow_missing=False)
        return cls(model, ctx)


    def get_model_details(self, sp_vec_file):
        data_csr, labels = load_svmlight_file(sp_vec_file)
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
        data_mat, labels = load_svmlight_file(sp_vec_file, n_features=len(self.vocab))
        return self.encode_data(data_mat, labels, use_probs=use_probs), labels

    def encode_texts(self, texts, use_probs=False):
        X, _ = self.vectorizer.transform(texts)
        return self.encode_data(X, None, use_probs=use_probs)

    def encode_data(self, data_mat, labels, use_probs=False):
        if isinstance(data_mat, scipy.sparse.csr.csr_matrix):
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
        x_size = data_to_iter.shape[0] * data_to_iter.shape[1]
        if x_size > MAX_DESIGN_MATRIX:
            logging.info("Sparse matrix has total size = {}. Using Sparse Matrix data batcher.".format(x_size))
            infer_iter = DataIterLoader(SparseMatrixDataIter(data_to_iter, covars, batch_size, last_batch_handle='discard', shuffle=False))
        else:
            infer_iter = DataIterLoader(mx.io.NDArrayIter(data_to_iter, covars,
                                                      batch_size, last_batch_handle='discard', shuffle=False))
        encodings = []
        for _, (data,labels) in enumerate(infer_iter):
            data = data.as_in_context(self.ctx)
            if self.covar_model and labels is not None:
                labels = labels.as_in_context(self.ctx)
                encs = self.model.encode_data_with_covariates(data, labels)
            else:
                encs = self.model.encode_data(data)
            if use_probs:
                e1 = encs - mx.nd.min(encs, axis=1).expand_dims(1)
                encs = mx.nd.softmax(e1 ** 0.5)
            encodings.extend(encs)
        ## handle the last batch explicitly as NDArrayIter doesn't do that for us
        if last_batch_size > 0:
            data = data_mat[-last_batch_size:].as_in_context(self.ctx)
            if self.covar_model and labels is not None:
                labels = mx.nd.one_hot(mx.nd.array(labels[-last_batch_size:], dtype='int'), self.n_covars).as_in_context(self.ctx)
                encs = self.model.encode_data_with_covariates(data, labels)
            else:
                encs = self.model.encode_data(data)
            if use_probs:
                #norm = mx.nd.norm(encs, axis=1, keepdims=True)
                e1 = encs - mx.nd.min(encs, axis=1).expand_dims(1)
                encs = mx.nd.softmax(e1 ** 0.5)
            encodings.extend(encs)
        return encodings


    def get_top_k_words_per_topic(self, k):
        sorted_ids = self.model.get_ordered_terms()
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
    


class SeqVEDInferencer(BaseInferencer):
    """Inferencer for sequence variational encoder-decoder models using BERT
    """
    def __init__(self, model, bert_vocab, max_length, ctx=mx.cpu()):
        super().__init__(ctx)
        self.model     = model
        self.bert_base = model.encoder
        self.tokenizer = BERTTokenizer(bert_vocab)
        self.transform = BERTSentenceTransform(self.tokenizer, max_length, pair=False)


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
        bert_base, vocab = nlp.model.get_model('bert_12_768_12',  
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False) #, output_attention=True)
        latent_dist = config['latent_distribution']['dist_type']       
        n_latent    = config['n_latent']
        kappa       = config['latent_distribution']['kappa']
        pad_id      = vocab[vocab.padding_token]
        max_sent_len = config['sent_size']  
        model = BertBowVED(bert_base, bow_vocab, latent_distrib=latent_dist,
                                n_latent=n_latent,
                                kappa = kappa,
                                batch_size=1)
        model.load_parameters(str(param_file), allow_missing=False, ignore_extra=True)
        return cls(model, vocab, max_length, ctx)


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
        return tokens, mx.nd.array([ids], dtype='int32'), mx.nd.array([lens], dtype='float32'), mx.nd.array([segs], dtype='int32')
    

    def encode_text(self, txt):                   
        tokens, ids, lens, segs = self.prep_text(txt)
        _, enc = self.model.encoder(ids.as_in_context(self.ctx),
                                              segs.as_in_context(self.ctx), lens.as_in_context(self.ctx))
        topic_encoding = self.model.latent_dist.get_mu_encoding(enc)
        return topic_encoding, tokens
