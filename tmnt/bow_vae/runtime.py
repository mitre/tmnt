# coding: utf-8

import json
import mxnet as mx
import gluonnlp as nlp
from mxnet import gluon
from mxnet.gluon import HybridBlock
from bow_models import BowNTM
from bow_doc_loader import collect_stream_as_sparse_matrix, DataIterLoader

class BowNTMInference(object):

    def __init__(self, param_file, specs_file, vocab_file, ctx=mx.cpu()):
        with open(specs_file) as f:
            specs = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        self.vocab = nlp.Vocab.from_json(voc_js)
        self.ctx = ctx
        self.n_latent = specs['n_latent']
        enc_dim = specs['enc_dim']
        
        gen_layers = specs['gen_layers']
        self.model = BowNTM(len(self.vocab), enc_dim, self.n_latent, gen_layers, ctx=ctx)
        self.model.load_parameters(param_file)
        

    def encode_texts(self, intexts):
        in_strms = [nlp.data.SimpleDataStream([t]) for t in intexts]
        strm = nlp.data.SimpleDataStream(in_strms)
        csr, _, _ = collect_stream_as_sparse_matrix(strm, pre_vocab=self.vocab)
        infer_iter = DataIterLoader(mx.io.NDArrayIter(csr, None, csr.shape[0], last_batch_handle='discard', shuffle=False))
        encodings = []
        for _, (data,_) in enumerate(infer_iter):
            data = data.as_in_context(self.ctx)
            encodings.append(self.model.encode_data(data))
        return encodings

    def get_top_k_words_per_topic(self, k):
        w = self.model.decoder.collect_params().get('weight').data()
        sorted_ids = w.argsort(axis=0, is_ascend=False)
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
            topic_terms.append(top_k)
        return topic_terms
            
