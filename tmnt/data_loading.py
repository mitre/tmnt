# coding: utf-8
# Copyright (c) 2019-2021. The MITRE Corporation.
"""
File/module contains routines for loading in text documents to sparse matrix representations
for efficient neural variational model training.
"""

import io
import itertools
import os
import logging
import scipy
import numpy as np
import string
import re
import json
from collections import OrderedDict, Counter
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle as sk_shuffle
from tmnt.preprocess.vectorizer import TMNTVectorizer
import random

from scipy import sparse as sp
from typing import List, Tuple, Dict, Optional, Union, NoReturn

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler
from torchtext.vocab import vocab as build_vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel, DistilBertTokenizer, BertModel, DistilBertModel, OpenAIGPTModel
from sklearn.model_selection import StratifiedKFold

#### Huggingface LLM-specific dataloading ####

llm_catalog = {
    'distilbert-base-uncased': (DistilBertTokenizer.from_pretrained, DistilBertModel.from_pretrained),
    'bert-base-uncased' : (AutoTokenizer.from_pretrained, BertModel.from_pretrained),
    'openai-gpt' : (AutoTokenizer.from_pretrained, OpenAIGPTModel.from_pretrained), 
    'sentence-transformers/all-mpnet-base-v2' : (AutoTokenizer.from_pretrained, AutoModel.from_pretrained)
    ## add more model options here if desired
    }

def get_llm(model_name):
    tok_fn, model_fn = llm_catalog[model_name]
    return tok_fn(model_name), model_fn(model_name)

def get_llm_tokenizer(model_name):
    tok_fn, _ = llm_catalog[model_name]
    return tok_fn(model_name)

def get_llm_model(model_name):
    _, model_fn = llm_catalog[model_name]
    return model_fn(model_name)

def get_unwrapped_llm_dataloader(data, bow_vectorizer, llm_name, label_map, batch_size, max_len, shuffle=False, device='cpu'):
    label_pipeline = lambda x: label_map.get(x, 0)
    text_pipeline  = get_llm_tokenizer(llm_name)
    
    def collate_batch(batch):
        label_list, text_list, mask_list, bow_list = [], [], [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            tokenized_result = text_pipeline(_text, return_tensors='pt', padding='max_length',
                                           max_length=max_len, truncation=True)
            bag_of_words,_ = bow_vectorizer.transform([_text])
            processed_text = tokenized_result['input_ids']
            mask = tokenized_result['attention_mask']
            mask_list.append(mask)
            text_list.append(processed_text)
            bow_list.append(bag_of_words)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list  = torch.vstack(text_list)
        mask_list  = torch.vstack(mask_list)
        bow_list   = torch.vstack([ sparse_coo_to_tensor(bow_vec.tocoo()) for bow_vec in bow_list ])
        return label_list.to(device), text_list.to(device), mask_list.to(device), bow_list.to(device)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)

def get_llm_dataloader(data, bow_vectorizer, llm_name, label_map, batch_size, max_len, shuffle=False, device='cpu'):
    return SingletonWrapperLoader(get_unwrapped_llm_dataloader(data, bow_vectorizer, llm_name, label_map, batch_size, max_len, shuffle=shuffle, device=device))


def get_llm_paired_dataloader(data_a, data_b, bow_vectorizer, llm_name, label_map, batch_size, max_len_a, max_len_b, 
                              shuffle_both=False, shuffle_a_only=True, device='cpu'):
    loader_a = get_unwrapped_llm_dataloader(data_a, bow_vectorizer, llm_name, label_map, batch_size, max_len_a, 
                                            shuffle=(shuffle_both or shuffle_a_only),
                                            device=device)   
    loader_b = get_unwrapped_llm_dataloader(data_b, bow_vectorizer, llm_name, label_map, batch_size, max_len_b, shuffle=shuffle_both,
                                            device=device) 
    return PairedDataLoader(loader_a, loader_b)


class StratifiedPairedLLMLoader():

    def __init__(self, data_a, data_b, bow_vectorizer, llm_name, label_map, batch_size, max_len_a, max_len_b, num_batches=0, device='cpu'):
        self.data_a = data_a
        self.data_b = data_b
        self.bow_vectorizer = bow_vectorizer
        self.llm_name = llm_name
        self.label_map = label_map
        self.batch_size = batch_size
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.device = device
        self.num_batches = num_batches or max(len(data_a), len(data_b)) // batch_size
        self.stratified_sampler = StratifiedDualBatchSampler(np.array([label_map[l] for (l,_) in data_a]), 
                                                             np.array([label_map[l] for (l,_) in data_b]),
                                                             batch_size,
                                                             self.num_batches)
        self.iterator = None
        self.label_pipeline = lambda x: label_map.get(x, 0)
        self.text_pipeline  = get_llm_tokenizer(llm_name)          
    
    def __iter__(self):
        self.iterator = iter(self.stratified_sampler)
        return self
    
    def __len__(self):
        return self.num_batches
    
    def __next__(self):
        indices_a, indices_b = next(self.iterator)
        batch_a = self._collate_batch([self.data_a[i_a] for i_a in indices_a], self.max_len_a)
        batch_b = self._collate_batch([self.data_b[i_b] for i_b in indices_b], self.max_len_b)
        return batch_a, batch_b 
    
    def _collate_batch(self, batch, max_len):
        label_list, text_list, mask_list, bow_list = [], [], [], []
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            tokenized_result = self.text_pipeline(_text, return_tensors='pt', padding='max_length',
                                           max_length=max_len, truncation=True)
            bag_of_words,_ = self.bow_vectorizer.transform([_text])
            processed_text = tokenized_result['input_ids']
            mask = tokenized_result['attention_mask']
            mask_list.append(mask)
            text_list.append(processed_text)
            bow_list.append(bag_of_words)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list  = torch.vstack(text_list)
        mask_list  = torch.vstack(mask_list)
        bow_list   = torch.vstack([ sparse_coo_to_tensor(bow_vec.tocoo()) for bow_vec in bow_list ])
        return label_list.to(self.device), text_list.to(self.device), mask_list.to(self.device), bow_list.to(self.device)






#def get_llm_paired_stratified_dataloader(data_a, data_b, bow_vectorizer, llm_name, label_map, batch_size, max_len_a, max_len_b, device='cpu'):
#    return 

##############################################

def to_label_matrix(yvs, num_labels=0):
    """Convert [(id1, id2, ...), (id1,id2,...) ... ] to Numpy matrix with multi-labels
    """
    if num_labels == 0:
        mx_val = 0
        for yi in yvs:
            for v in yi:
                if v > mx_val:
                    mx_val = v
        num_labels = int(mx_val + 1)
    li = []
    for yi in yvs:
        a = np.zeros(num_labels)
        a[np.array(yi, dtype='int64')] = 1
        li.append(a)
    return np.array(li), num_labels


class SparseDataset():
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, 
                 data:Union[np.ndarray, sp.coo_matrix, sp.csr_matrix], 
                 targets: Optional[Union[np.ndarray, sp.coo_matrix, sp.csr_matrix]]):
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == sp.coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            
        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == sp.coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets
        
    def __getitem__(self, index):
        targets_i = self.targets[index] if self.targets is not None else None
        return self.data[index], targets_i

    def __len__(self):
        return self.data.shape[0]
      
def sparse_coo_to_tensor(coo: sp.coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = (coo.row, coo.col)
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)


def sparse_batch_collate(batch): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    data_batch, targets_batch = batch[0]
    
    if type(data_batch[0]) == sp.csr_matrix:
        data_batch = data_batch.tocoo() # removed vstack
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if targets_batch is not None:
        if type(targets_batch[0]) == sp.csr_matrix:
            targets_batch = targets_batch.tocoo() # removed vstack
            targets_batch = sparse_coo_to_tensor(targets_batch)
        else:
            targets_batch = torch.LongTensor(targets_batch)
    return data_batch, targets_batch


class SparseDataLoader(DataLoader):

    def __init__(self, 
                 X: Union[sp.csr_matrix, sp.coo_matrix], y: np.array, shuffle=False, drop_last=False,
                 batch_size=1024, device='cpu'):
        self.batch_size = batch_size
        ds = SparseDataset(X, y)
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(ds,
                                                   generator=torch.Generator(device=device)),
            batch_size=batch_size,
            drop_last=False)
        super().__init__(ds, batch_size=1, collate_fn=sparse_batch_collate, generator=torch.Generator(device=device),
                         sampler=sampler, drop_last=drop_last)
        


class SingletonWrapperLoader():

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter   = iter(data_loader)

    def __iter__(self):
        self.data_iter = iter(self.data_loader)
        return self

    def __len__(self):
        return len(self.data_iter)

    def __next__(self):
        batch = self.data_iter.__next__()
        return (batch,)

    def next(self):
        return self.__next__()    


class PairedDataLoader():
    
    def __init__(self, data_loader1, data_loader2):
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2
        self.data_iter1   = iter(data_loader1)
        self.data_iter2   = iter(data_loader2) if data_loader2 is not None else None
        self.batch_index = 0
        self.end1         = False
        self.end2         = False


    def __iter__(self):
        self.data_iter1 = iter(self.data_loader1)
        self.data_iter2 = iter(self.data_loader2) if self.data_loader2 is not None else None
        self.batch_index = 0
        self.end1 = False
        self.end2 = False
        return self


    def __len__(self):
        if self.data_loader2 is not None:
            return max(len(self.data_loader1), len(self.data_loader2))
        else:
            return len(self.data_loader1)

    def __next__(self):
        try:
            batch1 = self.data_iter1.__next__()
        except StopIteration:
            if self.end2 or self.data_loader2 is None:
                raise StopIteration
            self.data_iter1 = iter(self.data_loader1)
            self.end1 = True
            batch1 = self.data_iter1.__next__()
        if self.data_loader2 is not None:
            try:
                batch2 = self.data_iter2.__next__()
            except StopIteration:
                if self.end1:
                    raise StopIteration
                self.data_iter2 = iter(self.data_loader2)
                self.end2 = True
                batch2 = self.data_iter2.__next__()
        else:
            batch2 = None
        return batch1, batch2

    def next(self):
        return self.__next__()



class RoundRobinDataLoader():
    
    def __init__(self, data_loaders):
        self.num_loaders = len(data_loaders)
        self.data_loaders = data_loaders
        self.data_iters = [iter(d) for d in data_loaders]
        self.data_totals = None
        self.ratio_remaining = np.array([1.0 for _ in data_loaders])

    def _get_iter_length(self, it):
        c = 0
        try:
            while True:
                _ = next(it)
                c += 1
        except:
            return c

    def _set_lengths(self, iters):
        self.data_totals = [ self._get_iter_length(it) for it in iters ]
        
    def __iter__(self):
        self._set_lengths( [iter(d) for d in self.data_loaders] )
        self.ratio_remaining[:] = 1.0
        self.data_iters = [iter(d) for d in self.data_loaders]
        return self

    def __len__(self):
        return sum([len(it) for it in self.data_iters])

    def __next__(self):
        it_id = np.argsort(-self.ratio_remaining)[0] ## get iterator with most elements left
        it = self.data_iters[it_id]
        batch = it.__next__()
        self.ratio_remaining[it_id] = ((self.ratio_remaining[it_id] * self.data_totals[it_id]) - 1) / self.data_totals[it_id]
        return batch

    def next(self):
        return self.__next__()
    


def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []
    data = OrderedDict([(default_name, data)]) # pylint: disable=redefined-variable-type
    return list(data.items())
    

def load_vocab(vocab_file, encoding='utf-8'):
    """
    Load a pre-derived vocabulary, assumes format consisting of a single word on each line.
    Note: this is a bit of a hack to use a counter to sort the vocab items IN THE ORDER THEY ARE FOUND IN THE FILE.
    """
    w_dict = {}
    words = []
    with io.open(vocab_file, 'r', encoding=encoding) as fp:
        for line in fp:
            els = line.split(' ')
            words.append(els[0].strip())
    ln_wds = len(words)
    for i in range(ln_wds):
        w_dict[words[i]] = ln_wds - i
    return build_vocab(w_dict)


def file_to_data(sp_file, voc_size, batch_size=1000):
    with open(sp_file) as f:
        for i, l in enumerate(f):
            pass
    data_size = i+1
    num_batches = data_size // batch_size
    last_batch_size = data_size % batch_size
    X, y = load_svmlight_file(sp_file, n_features=voc_size, dtype='int32', zero_based=True)
    wd_freqs = mx.nd.array(np.array(X.sum(axis=0)).squeeze())
    total_words = X.sum()
    return X, y, wd_freqs, total_words


class StratifiedDualBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y_a, y_b, batch_size, num_batches, shuffle=True):
        assert len(y_a.shape) == 1 # 'label array must be 1D'
        assert len(y_b.shape) == 1 
        self.y_a = y_a
        self.y_b = y_b
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.counts_a = Counter(y_a) 
        self.counts_b = Counter(y_b)
        self.class_weights_a = [0] * (max(np.max(y_a), np.max(y_b)) + 1)
        self.class_weights_b = [0] * (max(np.max(y_a), np.max(y_b)) + 1)
        for k in self.counts_a:
            self.class_weights_a[k] = self.counts_a[k] / len(y_a)
        for k in self.counts_b:
            self.class_weights_b[k] = self.counts_b[k] / len(y_b)
        self.class_indices_a = [0] * (max(np.max(y_a), np.max(y_b)) + 1)
        self.class_indices_b = [0] * (max(np.max(y_b), np.max(y_a)) + 1)
        for i in range(len(self.class_indices_a)):
            self.class_indices_a[i] = list(np.where(y_a == i)[0])
        for i in range(len(self.class_indices_b)):
            self.class_indices_b[i] = list(np.where(y_b == i)[0])
        self.a_only = self.counts_a.keys() - self.counts_b.keys()
        self.b_only = self.counts_b.keys() - self.counts_a.keys()
        self.use_with_replacement = (self.batch_size > len(self.class_weights_a))
        
    def _pop_leave_last(self, li):
        if len(li) == 1:
            return li[0]
        else:
            return li.pop()

    def __iter__(self):
        samplers_a = [ iter(RandomSampler(self.class_indices_a[c], replacement=True, num_samples=(self.num_batches * self.batch_size))) 
                      for c in range(len(self.class_indices_a)) ]
        samplers_b = [ iter(RandomSampler(self.class_indices_b[c], replacement=True, num_samples=(self.num_batches * self.batch_size))) 
                      for c in range(len(self.class_indices_b)) ]
        for i in range(self.num_batches):
            if i % 2 == 0:
                classes_a = list(WeightedRandomSampler(self.class_weights_a, self.batch_size, replacement=self.use_with_replacement))
                b_list = list(self.counts_b)
                random.shuffle(b_list)
                classes_b = [ self._pop_leave_last(b_list) if a in self.a_only else a for a in classes_a]
                batch_indices_a = [ self.class_indices_a[c][next(samplers_a[c])] for c in classes_a]                                 
                batch_indices_b = [ self.class_indices_b[c][next(samplers_b[c])] for c in classes_b]
            else:
                classes_b = list(WeightedRandomSampler(self.class_weights_b, self.batch_size, replacement=self.use_with_replacement))
                a_list = list(self.counts_a)
                random.shuffle(a_list)
                classes_a = [ self._pop_leave_last(a_list) if b in self.b_only else b for b in classes_b]
                batch_indices_a = [ self.class_indices_a[c][next(samplers_a[c])] for c in classes_a]
                batch_indices_b = [ self.class_indices_b[c][next(samplers_b[c])] for c in classes_b]                                 
            yield (batch_indices_a, batch_indices_b) 
                
    def __len__(self):
        return len(self.num_batches)



































