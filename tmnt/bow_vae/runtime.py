# coding: utf-8

import json
import mxnet as mx
import gluonnlp as nlp
from tmnt.bow_vae.bow_models import BowNTM
from tmnt.bow_vae.bow_doc_loader import collect_stream_as_sparse_matrix, DataIterLoader, BowDataSet

class BowNTMInference(object):

    def __init__(self, param_file, specs_file, vocab_file, ctx=mx.cpu()):
        self.max_batch_size = 2
        with open(specs_file) as f:
            specs = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        self.vocab = nlp.Vocab.from_json(voc_js)
        self.ctx = ctx
        self.n_latent = specs['n_latent']
        enc_dim = specs['enc_dim']
        gen_layers = specs['gen_layers']
        self.model = BowNTM(self.vocab, enc_dim, self.n_latent, ctx=ctx)
        self.model.load_parameters(str(param_file))
        

    def encode_texts(self, intexts):
        """
        intexts - should be a list of lists of tokens (each token list being a document)
        """
        in_strms = [nlp.data.SimpleDataStream([t]) for t in intexts]
        strm = nlp.data.SimpleDataStream(in_strms)
        return self.encode_text_stream(strm)

    def encode_text_stream(self, strm):
        csr, _, _ = collect_stream_as_sparse_matrix(strm, pre_vocab=self.vocab)
        batch_size = min(csr.shape[0], self.max_batch_size)
        last_batch_size = csr.shape[0] % batch_size        
        infer_iter = DataIterLoader(mx.io.NDArrayIter(csr[:-last_batch_size], None, batch_size, last_batch_handle='discard', shuffle=False))
        encodings = []
        for _, (data,_) in enumerate(infer_iter):
            data = data.as_in_context(self.ctx)
            encodings.extend(self.model.encode_data(data))
        ## handle the last batch explicitly as NDArrayIter doesn't do that for us
        if last_batch_size > 0:
            data = csr[-last_batch_size:].as_in_context(self.ctx)
            encodings.extend(self.model.encode_data(data))
        return encodings

    def get_top_k_words_per_topic(self, k):
        w = self.model.decoder.collect_params().get('weight').data()
        sorted_ids = w.argsort(axis=0, is_ascend=False)
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
            topic_terms.append(top_k)
        return topic_terms

    def _test_inference_on_directory(self, directory, file_pattern=None):
        """
        Temporary test method to demonstrate use of inference on a set of files in a directory
        """
        pat = '*.txt' if file_pattern is None else file_pattern
        dataset_strm = BowDataSet(directory, pat, sampler='sequential') # preserve file ordering
        return self.encode_text_stream(dataset_strm)
        
