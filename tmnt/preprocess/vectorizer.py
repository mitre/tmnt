# coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""


import io
import os
import json
import gluonnlp as nlp
import glob
from gluonnlp.data import Counter
from multiprocessing import Pool, cpu_count
from mantichora import mantichora
from atpbar import atpbar
import threading
import logging
import threading
import scipy.sparse as sp
import numpy as np
from queue import Queue
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file
from tmnt.preprocess import BasicTokenizer

__all__ = ['TMNTVectorizer']


class TMNTVectorizer(object):

    def __init__(self, custom_stop_word_file=None, text_key='body', label_key=None, min_doc_size=1, label_prefix=-1,
                 json_out_dir=None, vocab_size=2000, file_pat = '*.json', encoding='utf-8', initial_vocabulary=None):
        self.encoding = encoding
        self.text_key = text_key
        self.label_key = label_key
        self.label_prefix = label_prefix
        self.min_doc_size = min_doc_size
        self.json_rewrite = json_out_dir is not None
        self.json_out_dir = json_out_dir
        self.vocab = initial_vocabulary
        self.file_pat = file_pat
        self.vocab_size = vocab_size if initial_vocabulary is None else len(initial_vocabulary)
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=self.vocab_size,
                                          stop_words='english',
                                          vocabulary=(initial_vocabulary.token_to_idx if initial_vocabulary else None))
        self.label_map = {}

    @classmethod
    def from_vocab_file(cls, vocab_file):
        with io.open(vocab_file, 'r') as fp:
            voc_js = fp.read()
        return cls(initial_vocabulary=nlp.Vocab.from_json(voc_js))

    def _get_update_label_index(self, v):
        if self.label_prefix > 0:
            v = v[:self.label_prefix]
        i = self.label_map.get(v)
        if i is None:
            i = len(self.label_map)
            self.label_map[v] = i
        return i
    
    
    def get_vocab(self):
        if self.vocab is not None:
            return self.vocab
        else:
            vocab = nlp.Vocab({v: 1 for v in self.vectorizer.vocabulary_}, unknown_token=None, padding_token=None,
                              bos_token=None, eos_token=None)
            self.vocab = vocab
        return vocab
    
    def _tr_json(self, tr_method, json_file):
        fp = io.open(json_file, 'r', encoding=self.encoding)
        gen = ( json.loads(l)[self.text_key] for l in fp )
        rr = tr_method(gen)
        fp.close()
        return rr

    def _tr_json_dir(self, tr_method, json_dir):
        fps = [ io.open(ff, 'r', encoding=self.encoding) for ff in glob.glob(json_dir + '/' + self.file_pat) ]
        gen = (json.loads(l)[self.text_key] for fp in fps for l in fp)
        rr = tr_method(gen)
        for fp in fps:
            fp.close()
        return rr

    def _get_ys(self, json_file):
        if self.label_key is not None:
            ys = []
            with io.open(json_file, 'r', encoding=self.encoding) as fp:
                for j in fp:
                    label_string = json.loads(j)[self.label_key]
                    label_id = self._get_update_label_index(label_string)
                    ys.append(label_id)
            return ys
        else:
            return None

    def _get_ys_dir(self, json_dir):
        if self.label_key is not None:
            fps = [ ff for ff in glob.glob(json_dir + '/' + self.file_pat) ]
            ys = []
            for f in fps:
                yy = self._get_ys(f)
                ys.extend(yy)
            return ys
        else:
            return None

    def write_to_vec_file(self, X, y, vec_file):
        if y is None:
            y = np.zeros(X.shape[0])
        dump_svmlight_file(X, y, vec_file)

    def write_vocab(self, vocab_file):
        vocab = self.get_vocab()
        with io.open(vocab_file, 'w', encoding=self.encoding) as fp:
            for i in range(len(vocab.idx_to_token)):
                fp.write(vocab.idx_to_token[i])
                fp.write('\n')
                
    def transform(self, str_list):
        return self.vectorizer.transform(str_list), None

    def transform_json(self, json_file):
        X = self._tr_json(self.vectorizer.transform, json_file)
        y = self._get_ys(json_file)
        return X, y

    def transform_json_dir(self, json_dir):
        X = self._tr_json_dir(self.vectorizer.transform, json_dir)
        y = self._get_ys_dir(json_dir)
        return X, None

    def fit_transform(self, str_list):
        return self.vectorizer.fit_transform(str_list), None

    def fit_transform_json(self, json_file):
        X = self._tr_json(self.vectorizer.fit_transform, json_file)
        y = self._get_ys(json_file)
        return X, y

    def fit_transform_json_dir(self, json_dir):
        X = self._tr_json_dir(self.vectorizer.fit_transform, json_dir)
        y = self._get_ys_dir(json_dir)
        return X, y

        
    def _tr_in_place_json(self, tr_method, json_file):        
        X, _ = tr_method(json_file)
        json_path, file_name = os.path.split(json_file)
        js_path = self.json_out_dir if self.json_out_dir is not None else json_path
        n_json_file = os.path.join(js_path, "vec_"+file_name)
        if not os.path.exists(js_path):
            os.mkdir(js_path)
        with io.open(json_file, 'r', encoding = self.encoding) as fp:
            with io.open(n_json_file, 'w', encoding = self.encoding) as op:
                i = 0
                for l in fp:
                    js = json.loads(l)
                    _, inds, vls = sp.find(X[i])
                    i_inds = [int(a) for a in list(inds)]
                    i_vls = [float(a) for a in list(vls)]
                    js['sp_vec'] = list(zip(i_inds, i_vls))
                    op.write(json.dumps(js))
                    op.write('\n')

    def fit_transform_in_place_json(self, json_file):
        self._tr_in_place_json(self.fit_transform_json, json_file)

    def transform_in_place_json(self, json_file):
        self._tr_in_place_json(self.transform_json, json_file)
                    
