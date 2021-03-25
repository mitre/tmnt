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
import collections
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

    """Utility vectorizer that wraps `sklearn.feature_extraction.text.CountVectorizer` for use
    with TMNT dataset conventions.

    Parameters:
        text_key (str): Json key for text to use as document content
        label_key (str): Json key to use for label/covariate
        min_doc_size (int): Minimum number of tokens for inclusion in the dataset
        json_out_dir (str): Output directory for resulting JSON files when using inline JSON processing
        vocab_size (int): Number of vocabulary items (default=2000)
        file_pat (str): File pattern for input json files (default = *.json)
        encoding (str): Character encoding (default = 'utf-8')
        initial_vocabulary (str): Use existing vocabulary rather than deriving one from the data
        additional_feature_keys (list): List of strings for json keys that correspond to additional 
            features to use alongside vocabulary
        stop_word_file (str): Path to a file containing stop words (newline separated)
        label_min_cnt (int): Minimum number of occurrences of label; instance provided label id -1 if label occurs less than this value
        count_vectorizer_kwargs (dict): Dictionary of parameter values to pass to `sklearn.feature_extraction.text.CountVectorizer`
    """

    def __init__(self, text_key='body', label_key=None, min_doc_size=1, label_prefix=-1, label_remap=None,
                 json_out_dir=None, vocab_size=2000, file_pat = '*.json', encoding='utf-8', initial_vocabulary=None,
                 additional_feature_keys=None, stop_word_file=None, label_min_cnt=1,
                 count_vectorizer_kwargs={'max_df':0.95, 'min_df':2, 'stop_words':'english'}):
        self.encoding = encoding
        self.text_key = text_key
        self.label_key = label_key
        self.label_prefix = label_prefix
        self.label_remap = label_remap
        self.min_doc_size = min_doc_size
        self.json_rewrite = json_out_dir is not None
        self.json_out_dir = json_out_dir
        self.vocab = initial_vocabulary
        self.additional_feature_keys = additional_feature_keys
        self.file_pat = file_pat
        self.label_min_cnt = label_min_cnt
        self.vocab_size = vocab_size if initial_vocabulary is None else len(initial_vocabulary)
        self.cv_kwargs = self._update_count_vectorizer_args(count_vectorizer_kwargs, stop_word_file)
        if not 'token_pattern' in self.cv_kwargs:
            self.cv_kwargs['token_pattern'] = r'\b[A-Za-z][A-Za-z]+\b'
        self.vectorizer = CountVectorizer(max_features=self.vocab_size, 
                                          vocabulary=(initial_vocabulary.token_to_idx if initial_vocabulary else None),
                                          **self.cv_kwargs)
        self.label_map = {}

        
    def _update_count_vectorizer_args(self, cv_kwargs, stop_word_file):
        if stop_word_file:
            stop_words = self._get_stop_word_set(stop_word_file)
            cv_kwargs['stop_words'] = stop_words
        return cv_kwargs

    @classmethod
    def from_vocab_file(cls, vocab_file):
        with io.open(vocab_file, 'r') as fp:
            voc_js = fp.read()
        return cls(initial_vocabulary=nlp.Vocab.from_json(voc_js))

    def _get_stop_word_set(self, f):
        wds = []
        with io.open(f, 'r', encoding=self.encoding) as fp:
            for w in fp:
                wds.append(w)
        return list(set(wds))

    
    def get_vocab(self):
        if self.vocab is not None:
            return self.vocab
        else:
            tok_to_idx = self.vectorizer.vocabulary_
            cv_vocab = {v: 1 for v in tok_to_idx}
            cur_idx = len(tok_to_idx)
            if self.additional_feature_keys:
                if isinstance(self.additional_feature_keys, list):
                    for f in self.additional_feature_keys:
                        cv_vocab[f] = 1
                        tok_to_idx[f] = cur_idx
                        cur_idx += 1
                else:
                    ## assume it's a dictionary
                    for k in self.additional_feature_keys:
                        for v in self.additional_feature_keys[k]:
                            cv_vocab[k+':'+v] = 1
                            tok_to_idx[k+':'+v] = cur_idx
                            cur_idx += 1
            vocab = nlp.Vocab(cv_vocab, token_to_idx=tok_to_idx,
                              unknown_token=None, eos_token=None, bos_token=None, padding_token=None)
            self.vocab = vocab
        return vocab

    def _add_features_json(self, json_file, num_instances):
        if isinstance(self.additional_feature_keys, list):
            n_features = len(self.additional_feature_keys)
        else:
            n_features = 0
            for k in self.additional_feature_keys:
                n_features += len(self.additional_feature_keys[k])
        X_add = np.zeros((num_instances, n_features))
        with io.open(json_file, 'r', encoding=self.encoding) as fp:
            for i, l in enumerate(fp):
                js = json.loads(l)
                if isinstance(self.additional_feature_keys, list):
                    for j,feature in enumerate(self.additional_feature_keys):
                        X_add[i][j] = float(js[feature])
                else:
                    j = 0
                    for k in self.additional_feature_keys:
                        for feature in self.additional_feature_keys[k]:
                            X_add[i][j] = float(js[k][feature])
                            j += 1
        return sp.csr_matrix(X_add)

    def _add_features_json_dir(self, json_dir, num_instances):
        X_add = np.zeros((num_instances, len(self.additional_feature_keys)))
        fps = [ io.open(ff, 'r', encoding=self.encoding) for ff in glob.glob(json_dir + '/' + self.file_pat) ]
        for fp in fps:
            for i, l in enumerate(fp):
                js = json.loads(l)
                for j,feature in enumerate(self.additional_feature_keys):
                    v = float(js[feature])
                    X_add[i][j] = v
        for fp in fps:
            fp.close()
        return sp.csr_matrix(X_add)
        
    
    def _tr_json(self, tr_method, json_file):
        fp = io.open(json_file, 'r', encoding=self.encoding)
        gen = ( json.loads(l)[self.text_key] for l in fp )
        rr = tr_method(gen)
        if self.additional_feature_keys:
            X_add = self._add_features_json(json_file, rr.shape[0])
            rr = sp.csr_matrix(sp.hstack((rr, sp.csr_matrix(X_add))))
        fp.close()
        return rr

    def _tr_json_dir(self, tr_method, json_dir):
        fps = [ io.open(ff, 'r', encoding=self.encoding) for ff in glob.glob(json_dir + '/' + self.file_pat) ]
        gen = (json.loads(l)[self.text_key] for fp in fps for l in fp)
        rr = tr_method(gen)
        if self.additional_feature_keys:
            X_add = self._add_features_json_dir(json_dir, rr.shape[0])
            rr = sp.csr_matrix(sp.hstack((rr, sp.csr_matrix(X_add))))
        for fp in fps:
            fp.close()
        return rr

    def _get_y_strs(self, json_file):
        ys = []
        with io.open(json_file, 'r', encoding=self.encoding) as fp:
            for j in fp:
                label_string = json.loads(j)[self.label_key]
                if self.label_remap:
                    label_string = self.label_remap[label_string]
                ys.append(label_string)
        return ys

    def _get_y_strs_dir(self, json_dir):
        fps = [ ff for ff in glob.glob(json_dir + '/' + self.file_pat) ]
        ys = []
        for f in fps:
            yy = self._get_y_strs(f)
            ys.extend(yy)
        return ys

    def _get_y_ids(self, y_strs):
        fixed = len(self.label_map) > 1
        lab_map = self.label_map
        def _update(s):
            i = lab_map.get(s)
            if i is None:
                if not fixed:
                    i = len(lab_map)
                    lab_map[s] = i
                else:
                    i = -1
            return i
        cnts = collections.Counter(y_strs)
        y_ids = [ (_update(ys) if cnts[ys] >= self.label_min_cnt else -1) for ys in y_strs ]
        self.label_map = lab_map
        return y_ids

    def _get_ys(self, json_file):
        if self.label_key is not None:
            return self._get_y_ids(self._get_y_strs(json_file))
        else:
            return None

    def _get_ys_dir(self, json_dir):
        if self.label_key is not None:
            return self._get_y_ids(self._get_y_strs_dir(json_dir))
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
        return X, y

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
                    
