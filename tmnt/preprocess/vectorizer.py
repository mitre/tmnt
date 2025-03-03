# coding: utf-8
"""
Copyright (c) 2019-2021 The MITRE Corporation.
"""

import io
import os
import json
from tmnt.utils.vocab import Vocab, build_vocab
import glob
from multiprocessing import Pool, cpu_count
from mantichora import mantichora
from atpbar import atpbar
import collections
import scipy.sparse as sp
import numpy as np
from queue import Queue
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file
from tmnt.preprocess import BasicTokenizer
from typing import List, Dict, Optional, Any, Tuple
from collections import OrderedDict
from sklearn.utils import check_array
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS


__all__ = ['TMNTVectorizer', 'CTFIDFVectorizer']

class TMNTVectorizer(object):

    """
    Utility vectorizer that wraps :py:class:`sklearn.feature_extraction.text.CountVectorizer` for use
    with TMNT dataset conventions.

    Parameters:
        text_key: Json key for text to use as document content
        label_key: Json key to use for label/covariate
        min_doc_size: Minimum number of tokens for inclusion in the dataset
        label_remap: Dictionary mapping input label strings to alternative label set
        json_out_dir: Output directory for resulting JSON files when using inline JSON processing
        vocab_size: Number of vocabulary items (default=2000)
        file_pat: File pattern for input json files (default = '*.json')
        encoding: Character encoding (default = 'utf-8')
        initial_vocabulary: Use existing vocabulary rather than deriving one from the data
        additional_feature_keys: List of strings for json keys that correspond to additional 
                features to use alongside vocabulary
        stop_word_file: Path to a file containing stop words (newline separated)
        split_char: Single character string used to split label string into multiple labels 
                (for multilabel classification tasks)
        max_ws_tokens: Maximum number of (whitespace deliniated) tokens to use
        count_vectorizer_kwargs: Dictionary of parameter values to pass to 
                :py:class:`sklearn.feature_extraction.text.CountVectorizer`
    """
    def __init__(self, text_key: str = 'body', label_key: Optional[str] = None, min_doc_size: int = 1,
                 label_remap: Optional[Dict[str,str]] = None,
                 json_out_dir: Optional[str] = None, vocab_size: int = 2000, file_pat: str = '*.json',
                 encoding: str = 'utf-8', initial_vocabulary: Optional[Vocab] = None,
                 additional_feature_keys: List[str] = None, stop_word_file: str = None,
                 split_char: str = ',',
                 max_ws_tokens: int = -1,
                 source_key: Optional[str] = None,
                 source_json: Optional[str] = None,
                 count_vectorizer_kwargs: Dict[str, Any] = {'max_df':0.95, 'min_df':0.0, 'stop_words':'english'}):
        self.encoding = encoding
        self.max_ws_tokens = max_ws_tokens
        self.text_key = text_key
        self.label_key = label_key
        self.label_remap = label_remap
        self.split_char  = split_char
        self.min_doc_size = min_doc_size
        self.json_rewrite = json_out_dir is not None
        self.json_out_dir = json_out_dir
        self.vocab = initial_vocabulary
        self.additional_feature_keys = additional_feature_keys
        self.file_pat = file_pat
        self.vocab_size = vocab_size if initial_vocabulary is None else len(initial_vocabulary)
        self.cv_kwargs = self._update_count_vectorizer_args(count_vectorizer_kwargs, stop_word_file)
        if not 'token_pattern' in self.cv_kwargs:
            self.cv_kwargs['token_pattern'] = r'\b[A-Za-z][A-Za-z]+\b'
        if source_key and source_json:
            source_terms = self._get_source_specific_terms(source_json, 10, text_key, source_key, 
                                                           {'token_pattern': self.cv_kwargs['token_pattern'],
                                                           'stop_words': self.cv_kwargs['stop_words'],
                                                            'max_df': 1.0, 'min_df':0.0})
            stop_words = set(source_terms)
            stop_words.update(set(ENGLISH_STOP_WORDS))
            self.cv_kwargs['stop_words'] = frozenset(stop_words)
        self.vectorizer = CountVectorizer(max_features=self.vocab_size, 
                                          vocabulary=(initial_vocabulary.get_itos() if initial_vocabulary else None),
                                          **self.cv_kwargs)
        self.label_map = {}


    def _get_source_specific_terms(self, json_file, k: int, text_key: str, source_key: str, cv_kwargs):
        by_source = {}
        with io.open(json_file) as fp:
            for l in fp:
                js = json.loads(l)
                txt = js[text_key]
                src = js[source_key]
                if src not in by_source:
                    by_source[src] = []
                by_source[src].append(txt)
        docs_by_source = [''.join(txts) for txts in by_source.values()]
        print(cv_kwargs)
        count_vectorizer = CountVectorizer(**cv_kwargs)
        count = count_vectorizer.fit_transform(docs_by_source)
        ctfidf = CTFIDFVectorizer().fit_transform(count)
        tok_to_idx = list(count_vectorizer.vocabulary_.items())
        tok_to_idx.sort(key = lambda x: x[1])
        ordered_vocab = OrderedDict([ (k,1) for (k,_) in tok_to_idx ])
        ovocab = build_vocab(ordered_vocab)
        per_source_tokens = []
        for i in range(ctfidf.shape[0]):
            ts = ctfidf[i].toarray().squeeze()
            per_source_tokens.append(ovocab.lookup_tokens((-ts).argsort()[:k]))
        final_tokens_intersect = set(per_source_tokens[0])
        final_tokens_union = set(per_source_tokens[0])
        for src_tokens in per_source_tokens:
            final_tokens_intersect.intersection_update(src_tokens)
            final_tokens_union.update(src_tokens)
        res = final_tokens_union - final_tokens_intersect
        print("Removed terms = {}".format(res))
        return final_tokens_union - final_tokens_intersect


        
    def _update_count_vectorizer_args(self, cv_kwargs: Dict[str, Any], stop_word_file: str) -> Dict[str, Any]:
        if stop_word_file:
            stop_words = self._get_stop_word_set(stop_word_file)
            cv_kwargs['stop_words'] = stop_words
        return cv_kwargs

    @classmethod
    def from_vocab_file(cls, vocab_file: str) -> 'TMNTVectorizer':
        """Class method that creates a TMNTVectorizer from a vocab file

        Parameters:
            vocab_file: String to vocabulary file path.

        Returns:
            TMNTVectorizer
        """
        with io.open(vocab_file, 'r') as fp:
            voc_dict = json.loads(fp.read())
        return cls(initial_vocabulary=build_vocab(voc_dict))

    def _get_stop_word_set(self, f: str) -> List[str]:
        wds = []
        with io.open(f, 'r', encoding=self.encoding) as fp:
            for w in fp:
                wds.append(w.strip())
        return list(set(wds))

    
    def get_vocab(self) -> Vocab:
        """Returns the vocabulary associated with the vectorizer

        Returns:
            vocabulary
        """
        if self.vocab is not None:
            return self.vocab
        else:
            tok_to_idx = list(self.vectorizer.vocabulary_.items())
            tok_to_idx.sort(key = lambda x: x[1])
            ordered_vocab = [ (k,1) for (k,_) in tok_to_idx ]
            if self.additional_feature_keys:
                if isinstance(self.additional_feature_keys, list):
                    for f in self.additional_feature_keys:
                        ordered_vocab.append((f,1))
                else:
                    ## assume it's a dictionary
                    for k in self.additional_feature_keys:
                        for v in self.additional_feature_keys[k]:
                            ordered_vocab.append((k+':'+v, 1))
            cv_vocab = OrderedDict(ordered_vocab)                            
            vb = build_vocab(cv_vocab)
            self.vocab = vb
        return vb

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

    def _truncate_to_ws_tokens(self, s):
        if self.max_ws_tokens > 0:
            ns = ""
            toks = s.split(' ')
            for i in range(min(self.max_ws_tokens, len(toks))):
                ns += ' '
                ns += toks[i]
            return ns
        else:
            return s

    
    def _tr_json(self, tr_method, json_file):
        fp = io.open(json_file, 'r', encoding=self.encoding)
        gen = ( self._truncate_to_ws_tokens(json.loads(l)[self.text_key]) for l in fp )
        rr = tr_method(gen)
        if self.additional_feature_keys:
            X_add = self._add_features_json(json_file, rr.shape[0])
            rr = sp.csr_matrix(sp.hstack((rr, sp.csr_matrix(X_add))))
        fp.close()
        return rr

    def _tr_json_dir(self, tr_method, json_dir):
        fps = [ io.open(ff, 'r', encoding=self.encoding) for ff in glob.glob(json_dir + '/' + self.file_pat) ]
        gen = ( self._truncate_to_ws_tokens(json.loads(l)[self.text_key]) for fp in fps for l in fp)
        rr = tr_method(gen)
        if self.additional_feature_keys:
            X_add = self._add_features_json_dir(json_dir, rr.shape[0])
            rr = sp.csr_matrix(sp.hstack((rr, sp.csr_matrix(X_add))))
        for fp in fps:
            fp.close()
        return rr

    def _get_y_strs(self, json_file):
        ys = [] # ys will be a list of lists of strings to accomodate multilabel data
        with io.open(json_file, 'r', encoding=self.encoding) as fp:
            for j in fp:
                js = json.loads(j)
                label_string = js.get(self.label_key)
                label_string_list = label_string.split(self.split_char)
                if self.label_remap:
                    label_string_list = [ self.label_remap.get(label_string) or label_string for label_string in label_string_list ]
                ys.append(label_string_list) 
        return ys

    def _get_y_strs_dir(self, json_dir):
        fps = [ ff for ff in glob.glob(json_dir + '/' + self.file_pat) ]
        ys = []
        for f in fps:
            yy = self._get_y_strs(f)
            ys.extend(yy)
        return ys

    def _get_y_ids(self, y_strs):
        # y_strs is a list of lists of strings
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
        cnts = collections.Counter([s for yi in y_strs for s in yi])
        y_ids = [ [ _update(ys) for ys in y_str_list ] for y_str_list in y_strs ]
        max_ids_per_instance = max([ len(yi_s) for yi_s in y_ids ])
        if max_ids_per_instance == 1:
            y_ids = np.array([ i for yi in y_ids for i in yi ]) ## flatten if we only have single label classification (most situations)
        else:
            li = []
            for yi in y_ids:
                a = np.zeros(len(lab_map))
                a[np.array(yi, dtype='int64')] = 1.0
                li.append(a)
            y_ids = np.array(li)
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

    def write_to_vec_file(self, X: sp.csr.csr_matrix, y: Optional[np.ndarray], vec_file: str) -> None:
        """Write document-term matrix and optional label vector to file in svmlight format.
        
        Parameters:
            X: document-term (sparse) matrix
            y: optional label vector (or matrix for multilabel documents)
            vec_file: string denoting path to output vector file
        """
        if y is None:
            y = np.zeros(X.shape[0])
        multilabel = len(y.shape) > 1
        dump_svmlight_file(X, y, vec_file, multilabel=multilabel)

    def write_vocab(self, vocab_file: str) -> None:
        """Write vocabulary to disk.

        Parameters:
            vocab_file: Write out vocabulary to this file (one word per line)
        Returns:
            None
        """
        vocab = self.get_vocab()
        with io.open(vocab_file, 'w', encoding=self.encoding) as fp:
            for i in range(len(vocab.idx_to_token)):
                fp.write(vocab.idx_to_token[i])
                fp.write('\n')
                
    def transform(self, str_list: List[str]) -> Tuple[sp.csr.csr_matrix, None]:
        """Transforms a list of strings into a sparse matrix.

        Transforms a single json list file into a tuple, the first element of which is 
        a single sparse matrix **X** and the second element is always `None`.

        Parameters:
            str_list: List of document strings
        Returns:
            Tuple of X,None - sparse matrix of the input, second element is always None in this case
        """
        return self.vectorizer.transform(str_list), None

    def transform_json(self, json_file: str) -> Tuple[sp.csr.csr_matrix, Optional[np.ndarray]]:
        """Transforms a single json list file into matrix/vector format(s).         

        Transforms a single json list file into a tuple, the first element being a
        single sparse matrix **X** and the second an (optional) label vector **y**.

        Parameters:
            json_file: Input file containing one document per line in serialized json format
        Returns:
            Tuple containing sparse document-term matrix X and optional label vector y
        """
        X = self._tr_json(self.vectorizer.transform, json_file)
        y = self._get_ys(json_file)
        return X, y

    def transform_json_dir(self, json_dir: str) -> Tuple[sp.csr.csr_matrix, Optional[np.ndarray]]:
        """Transforms a the specified directory's json list files into matrix formats.

        Parameters:
            json_dir: A string denoting the path to a directory containing json list files to process
        Returns:
            Tuple containing sparse document-term matrix X and optional label vector y
        """
        X = self._tr_json_dir(self.vectorizer.transform, json_dir)
        y = self._get_ys_dir(json_dir)
        return X, y

    def fit_transform(self, str_list: List[str]) -> Tuple[sp.csr.csr_matrix, None]:
        """Learns a vocabulary and transforms the input into into matrix formats.

        As a side-effect, this function induces a vocabulary of the inputs.

        Parameters:
            str_list: List of document strings
        Returns:
            Tuple containing sparse document-term matrix X and optional label vector y
        """
        return self.vectorizer.fit_transform(str_list), None

    def fit_transform_json(self, json_file: str) -> Tuple[sp.csr.csr_matrix, Optional[np.ndarray]]:
        """Learns a vocabulary and transforms the input into into matrix formats.

        As a side-effect, this function induces a vocabulary of the inputs.

        Parameters:
            json_file: Input file containing one document per line in serialized json format
        Returns:
            Tuple containing sparse document-term matrix X and optional label vector y
        """
        X = self._tr_json(self.vectorizer.fit_transform, json_file)
        y = self._get_ys(json_file)
        return X, y

    def fit_transform_json_dir(self, json_dir: str) -> Tuple[sp.csr.csr_matrix, Optional[np.ndarray]]:
        """Learns a vocabulary and transforms the input into into matrix formats.

        As a side-effect, this function induces a vocabulary of the inputs.

        Parameters:
            json_dir: A string denoting the path to a directory containing json list files to process
        Returns:
            Tuple containing sparse document-term matrix X and optional label vector y
        """
        X = self._tr_json_dir(self.vectorizer.fit_transform, json_dir)
        y = self._get_ys_dir(json_dir)
        return X, y



class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)
        self._idf_diag = None

    def fit(self, X: sp.csr_matrix):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.

        """

        # Prepare input
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        # Calculate IDF scores
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        avg_nr_samples = int(X.sum(axis=1).mean())
        idf = np.log(avg_nr_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=dtype)
        setattr(self, 'idf_', True)
        return self

    def transform(self, X: sp.csr_matrix, copy=True) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)

        """

        # Prepare input
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        _, n_features = X.shape

        # idf_ being a property, the automatic attributes detection
        # does not work as usual and we need to specify the attribute
        # name:
        check_is_fitted(self, attributes=["idf_"],
                        msg='idf vector is not fitted')

        # Check if expected nr features is found
        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"
                             " has been trained with n_features=%d" % (
                                 n_features, expected_n_features))

        X = X * self._idf_diag

        if self.norm:
            X = normalize(X, axis=1, norm='l1', copy=False)

        return X