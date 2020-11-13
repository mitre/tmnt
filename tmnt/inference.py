# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import json
import mxnet as mx
import numpy as np
import gluonnlp as nlp
import io
import os
import scipy
from tmnt.modeling import BowVAEModel, MetaDataBowVAEModel, BertBowVED
from tmnt.data_loading import DataIterLoader, file_to_data
from tmnt.preprocess.tokenizer import BasicTokenizer
from tmnt.preprocess.vectorizer import TextVectorizer
from multiprocessing import Pool
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform



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

    def __init__(self, param_file=None, config_file=None, vocab_file=None, model_dir=None, ctx=mx.cpu()):
        super().__init__(ctx)
        self.max_batch_size = 2
        if model_dir is not None:
            param_file = os.path.join(model_dir,'model.params')
            vocab_file = os.path.join(model_dir,'vocab.json')
            config_file = os.path.join(model_dir,'model.config')
        with open(config_file) as f:
            config = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        self.vocab = nlp.Vocab.from_json(voc_js)
        self.vectorizer = TextVectorizer(min_doc_size=1)

        self.n_latent = config['n_latent']
        enc_dim = config['enc_hidden_dim']
        lat_distrib = config['latent_distribution']['dist_type']
        n_encoding_layers = config.get('num_enc_layers', 0)
        enc_dr= float(config.get('enc_dr', 0.0))
        emb_size = config['derived_info']['embedding_size']
        if 'n_covars' in config:
            self.covar_model = True
            self.n_covars = config['n_covars']
            self.label_map = config['l_map']
            self.covar_net_layers = config.get('covar_net_layers')
            self.model = MetaDataBowVAEModel(self.label_map, self.n_covars,
                                        self.vocab, enc_dim, self.n_latent, emb_size, latent_distrib=lat_distrib,
                                        n_encoding_layers=n_encoding_layers, enc_dr=enc_dr,                                        
                                        covar_net_layers = self.covar_net_layers, ctx=ctx)
        else:
            self.covar_model = False
            self.model = BowVAEModel(self.vocab, enc_dim, self.n_latent, emb_size, latent_distrib=lat_distrib,
                                n_encoding_layers=n_encoding_layers, enc_dr=enc_dr,
                                ctx=ctx)
        self.model.load_parameters(str(param_file), allow_missing=False)


    def get_model_details(self, sp_vec_file):
        data_csr, labels, _, _ = file_to_data(sp_vec_file, len(self.vocab))
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
    

    def export_full_model_inference_details(self, sp_vec_file, ofile):
        d = self.get_pyldavis_details(sp_vec_file)
        with io.open(ofile, 'w') as fp:
            json.dump(d, fp, sort_keys=True, indent=4)        


    def encode_texts(self, intexts):
        """
        intexts - should be a list of lists of tokens (each token list being a document)
        """
        in_strms = [nlp.data.SimpleDataStream([t]) for t in intexts]
        strm = nlp.data.SimpleDataStream(in_strms)
        return self.encode_text_stream(strm)

    def encode_vec_file(self, sp_vec_file):
        data_mat, labels, _, _ = file_to_data(sp_vec_file, len(self.vocab))
        return self.encode_data(data_mat, labels), labels

    def encode_text_stream(self, strm):
        vecs = [ self.vectorizer.vectorize_string(s, self.vocab) for s in strm ]
        data = mx.nd.array(vecs)
        return self.encode_data(data, None)

    def encode_data(self, data_mat, labels, use_probs=False):
        if isinstance(data_mat, scipy.sparse.csr.csr_matrix):
            data_mat = mx.nd.sparse.csr_matrix(data_mat, dtype='float32')
        batch_size = min(data_mat.shape[0], self.max_batch_size)
        last_batch_size = data_mat.shape[0] % batch_size
        covars = mx.nd.one_hot(mx.nd.array(labels, dtype='int'), self.n_covars) \
            if self.covar_model and labels[:-last_batch_size] is not None else None
        infer_iter = DataIterLoader(mx.io.NDArrayIter(data_mat[:-last_batch_size], covars,
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
        sorted_ids = self.get_ordered_terms()
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
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
    

class BaseTextEncoder(object):
    """Base text encoder for various topic models

    Args:
        inferencer (`tmnt.inference.BaseInferencer`): Inferencer object that runs the encoder portion of a VAE/VED model.
        use_probs (bool): Map topic vector encodings to the simplex (sum to 1).
        temperature (float): Temperature to sharpen/flatten encoding distribution.
    """
    def __init__(self, inferencer, use_probs=True, temperature=0.5):
        self.temp      = temperature
        self.inference = inference
        self.use_probs = use_probs

    def encode_single_string(self, txt_string):
        raise NotImplementedError

    def encode_batch(self, txtx, covars=None, pool_size=4):
        raise NotImplementedError



class BOWTextEncoder(object):

    """
    Takes a batch of text strings/documents and returns a matrix of their encodings (each row in the matrix
    corresponds to the encoding of the corresponding input text).

    Parameters
    ----------
    inference - the inference object using the trained model
    use_probs - boolean that indicates whether raw topic scores should be converted to probabilities or not (default = True)
    pool_size - integer that specifies the number of processes to use for concurrent text pre-processing
    """
    def __init__(self, inference, use_probs=True, temp=0.5):
        self.temp      = temp
        self.inference = inference
        self.use_probs = use_probs
        self.vocab_len = len(self.inference.vocab.idx_to_token)
        self.tokenizer = BasicTokenizer(do_lower_case=True, use_stop_words=False)

    def _txt_to_vec(self, txt):
        toks = self.tokenizer.tokenize(txt)
        ids = [self.inference.vocab[token] for token in toks if token in self.inference.vocab]
        return ids

    def encode_single_string(self, txt):
        data = mx.nd.zeros((1, self.vocab_len))
        for t in self._txt_to_vec(txt):
            data[0][t] += 1.0
        encs = self.inference.model.encode_data(data)
        if self.use_probs:
            e1 = encs - mx.nd.min(encs, axis=1).expand_dims(1)
            encs = mx.nd.softmax(e1 ** 0.5)
        return encs

    def encode_batch(self, txts, covars=None, pool_size=4):
        pool = Pool(pool_size)
        ids  = pool.map(self._txt_to_vec, txts)
        # Prevent zombie threads from hanging around
        pool.terminate()
        pool = None
        data = mx.nd.zeros((len(txts), self.vocab_len))
        for i, txt_ids in enumerate(ids):
            for t in txt_ids:
                data[i][t] += 1.0
        model = self.inference.model
        if covars:
            covar_ids = mx.nd.array([model.label_map[v] for v in covars])
            covars = mx.nd.one_hot(covar_ids, depth=len(model.label_map))
            encs = model.encode_data_with_covariates(data, covars)
        else:
            encs = model.encode_data(data)
        if self.use_probs:
            e1 = encs - mx.nd.min(encs, axis=1).expand_dims(1)
            encs = mx.nd.softmax(e1 ** self.temp)
        return encs
    

class SeqVEDInferencer(BaseInferencer):
    """Inferencer for sequence variational encoder-decoder models using BERT
    """
    def __init__(self, param_file=None, config_file=None, vocab_file=None, model_dir=None, ctx=mx.cpu()):
        super().__init__(ctx)
        if model_dir is not None:
            param_file = os.path.join(model_dir, 'model.params')
            vocab_file = os.path.join(model_dir, 'vocab.json')
            config_file = os.path.join(model_dir, 'model.config')
        with open(config_file) as f:
            config = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        self.bow_vocab = nlp.Vocab.from_json(voc_js)
        self.ctx = ctx
        self.bert_base, self.vocab = nlp.model.get_model('bert_12_768_12',  
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False) #, output_attention=True)
        self.latent_dist = config['latent_distribution']['dist_type']       
        self.n_latent    = config['n_latent']
        self.kappa       = config['latent_distribution']['kappa']
        self.pad_id      = self.vocab[self.vocab.padding_token]
        self.max_sent_len = config['sent_size']  
        self.model = BertBowVED(self.bert_base, self.bow_vocab, latent_distrib=self.latent_dist,
                                n_latent=self.n_latent,
                                kappa = self.kappa,
                                batch_size=1)
        self.tokenizer = BERTTokenizer(self.vocab)
        self.transform = BERTSentenceTransform(self.tokenizer, self.max_sent_len, pair=False)
        self.model.load_parameters(str(param_file), allow_missing=False, ignore_extra=True)


    def _embed_sequence(self, ids, segs):
        embeddings = self.bert_base.word_embed(ids) + self.bert_base.token_type_embed(segs)
        return embeddings.transpose((1,0,2))

    def _encode_from_embedding(self, embeddings, lens):
        outputs, _ = self.bert_base.encoder(embeddings, valid_length=lens)
        outputs = outputs.transpose((1,0,2))
        # outputs should be (batch, seq_len, C) shaped now
        pooled_out = self.bert_base._apply_pooling(outputs)
        topic_encoding = self.model.latent_dist.mu_encoder(pooled_out)
        return topic_encoding


    def prep_text(self, txt):    # used for integrated gradients
        tokens = self.tokenizer(txt)
        ids, lens, segs = self.transform((txt,))
        return tokens, mx.nd.array([ids], dtype='int32'), mx.nd.array([lens], dtype='int32'), mx.nd.array([segs], dtype='int32')
    

    def encode_text(self, txt):                   
        tokens, ids, lens, segs = self.prep_text(txt)
        _, enc = self.model.encoder(ids.as_in_context(self.ctx),
                                              segs.as_in_context(self.ctx), lens.astype('float32').as_in_context(self.ctx))
        topic_encoding = self.model.latent_dist.mu_encoder(enc)
        return topic_encoding, tokens
