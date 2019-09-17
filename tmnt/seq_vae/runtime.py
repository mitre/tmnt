# coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""

import json
import mxnet as mx
import gluonnlp as nlp
import io
from tmnt.seq_vae.tokenization import BasicTokenizer
from tmnt.seq_vae.trans_seq_models import PureTransformerVAE

class SeqVAEInference(object):

    def __init__(self, param_file, config_file, vocab_file, sent_size=None, ctx=mx.cpu()):
        self.max_batch_size = 2
        with open(config_file) as f:
            config = json.loads(f.read())
        with open(vocab_file) as f:
            voc_js = f.read()
        self.vocab = nlp.Vocab.from_json(voc_js)
        self.ctx = ctx
        self.latent_dist = config['latent_dist']
        self.num_units   = config['num_units']
        self.num_heads   = config['num_heads']        
        self.hidden_size = config['hidden_size']
        self.layers      = config['transformer_layers']
        self.n_latent    = config['n_latent']
        self.kappa       = config['kappa']
        self.embedding_size = config['embedding_size']
        self.pad_id      = self.vocab[self.vocab.padding_token]
        self.max_sent_len = sent_size if sent_size else config['sent_size']
        self.model = PureTransformerVAE(self.vocab, self.embedding_size, self.latent_dist, self.num_units, self.hidden_size, self.num_heads,
                                        self.n_latent, self.max_sent_len, self.layers, kappa=self.kappa, batch_size=1)
        self.tokenizer = BasicTokenizer(do_lower_case=True)
        self.model.load_parameters(str(param_file), allow_missing=False)


    def prep_text(self, txt):
        toks = self.tokenizer.tokenize(txt)[:(self.max_sent_len-2)]
        toks = ['<bos>'] + toks + ['<eos>']
        ids = []
        for t in toks:
            try:
                ids.append(self.vocab[t])
            except:
                ids.append(self.vocab[self.vocab.unknown_token])
            padded_ids = ids[:self.max_sent_len] if len(ids) >= self.max_sent_len else ids + [self.pad_id] * (self.max_sent_len - len(ids))
        return mx.nd.array(padded_ids, dtype='int').expand_dims(0)


    def encode_text(self, txt):
        ids = self.prep_text(txt)
        _, _, _, predictions = self.model(ids)
        reconstructed_sent_ids = mx.nd.argmax(predictions[0],1).asnumpy()
        rec_sent = [self.vocab.idx_to_token[int(i)] for i in reconstructed_sent_ids if i != self.pad_id]   # remove <PAD> token from rendering
        return rec_sent

        
                         
