# coding: utf-8

import io
import json
import gluonnlp as nlp
import glob
from gluonnlp.data import Counter
from multiprocessing import Pool, cpu_count
import threading

from tmnt.preprocess import BasicTokenizer

__all__ = ['get_sparse_vecs', 'set_text_key', 'set_min_doc_size']

tokenizer = BasicTokenizer(use_stop_words=True)
txt_key = 'text'
min_doc_size = 6

def set_text_key(k):
    global txt_key
    txt_key = k

def set_min_doc_size(k):
    global min_doc_size
    min_doc_size = k

def get_counter_file(json_file):
    counter = None
    with io.open(json_file, 'r') as fp:
        for l in fp:
            js = json.loads(l)
            txt = js[txt_key] ## text field
            counter = nlp.data.count_tokens(tokenizer.tokenize(txt), counter = counter)
    return counter

def sp_fn(json_file_and_vocab):
    json_file, vocab = json_file_and_vocab
    sp_vecs = []
    with io.open(json_file, 'r') as fp:
        for l in fp:
            js = json.loads(l)
            toks = tokenizer.tokenize(js[txt_key])
            tok_ids = [vocab[token] for token in toks if token in vocab]
            if (len(tok_ids) >= min_doc_size):
                cnts = nlp.data.count_tokens(tok_ids)
                sp_vecs.append(sorted(cnts.items()))
    return sp_vecs

def get_counter_dir_parallel(json_dir, pat='*.json'):
    files = glob.glob(json_dir + '/' + pat)
    if len(files) > 2:
        p = Pool(cpu_count())
        counters = p.map(get_counter_file, files)
    else:
        counters = map(get_counter_file, files)
    count = sum(counters, Counter())
    return count

def get_vocab(counter, size=2000):
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                              bos_token=None, eos_token=None, min_freq=5, max_size=size)
    return vocab


def get_sparse_vecs(sp_out_file, vocab_out_file, json_dir, vocab_size=2000, i_vocab=None, full_histogram_file=None, pat='*.json'):
    files = glob.glob(json_dir + '/' + pat)
    if i_vocab is None:
        counter = get_counter_dir_parallel(json_dir, pat)
        vocab = get_vocab(counter, vocab_size)
    else:
        vocab = i_vocab
    files_and_vocab = [(f,vocab) for f in files]
    if len(files) > 2:
        p = Pool(cpu_count())
        sp_vecs = p.map(sp_fn, files_and_vocab)
    else:
        sp_vecs = map(sp_fn, files_and_vocab)
    ## write sp vecs
    with io.open(sp_out_file, 'w') as fp:
        for block in sp_vecs:
            for v in block:
                fp.write('-1')  ## the label (-1  if none)
                for (i,c) in v:
                    fp.write(' ')
                    fp.write(str(i))
                    fp.write(':')
                    fp.write(str(c))
                fp.write('\n')
    if i_vocab is None: ## print out vocab if we had to create it
        with io.open(vocab_out_file, 'w') as fp:
            for i in range(len(vocab.idx_to_token)):
                fp.write(vocab.idx_to_token[i])
                fp.write('\n')
    if full_histogram_file:
        with io.open(full_histogram_file, 'w') as fp:
            items = list(counter.items())
            items.sort(key=lambda x: -x[1])
            for k,v in items:
                fp.write(str(k))
                fp.write(' ')
                fp.write(str(v))
                fp.write('\n')
    return vocab



