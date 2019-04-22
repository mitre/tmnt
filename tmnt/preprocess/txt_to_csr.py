# coding: utf-8

import io
import json
import gluonnlp as nlp
import glob
from gluonnlp.data import Counter
from multiprocessing import Pool, cpu_count
import threading

from tmnt.preprocess import BasicTokenizer

__all__ = ['get_sparse_vecs_txt']

tokenizer = BasicTokenizer(use_stop_words=True)

def get_counter_file_batch(txt_file_batch):
    counter = Counter()
    for txt_file in txt_file_batch:
        with io.open(txt_file, 'r') as fp:
            for txt in fp:
                counter = nlp.data.count_tokens(tokenizer.tokenize(txt), counter = counter)
    return counter

def sp_fn(txt_file_and_vocab):
    txt_file, vocab = txt_file_and_vocab
    sp_vecs = []
    with io.open(txt_file, 'r') as fp:
        doc_tok_ids = []
        for txt in fp:
            toks = tokenizer.tokenize(txt)
            tok_ids = [vocab[token] for token in toks if token in vocab]
            doc_tok_ids.extend(tok_ids)
        cnts = nlp.data.count_tokens(doc_tok_ids)
        sp_vecs.append(sorted(cnts.items()))
    return sp_vecs

def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def get_counter_dir_parallel(txt_dir, pat='*.txt'):
    files = glob.glob(txt_dir + '/' + pat)
    batch_size = max(1, int(len(files) / 20))
    file_batches = list(batches(files, batch_size))
    if len(file_batches) > 2:
        p = Pool(cpu_count())
        counters = p.map(get_counter_file_batch, file_batches)
    else:
        counters = map(get_counter_file_batch, file_batches)
    print("Built {} counters".format(len(counters)))
    return sum(counters, Counter())

def get_vocab(counter, size=2000):
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                              bos_token=None, eos_token=None, min_freq=5, max_size=size)
    return vocab

def get_sparse_vecs_txt(sp_out_file, vocab_out_file, txt_dir, vocab_size=2000, i_vocab=None, full_histogram_file=None, pat='*.txt'):
    files = glob.glob(txt_dir + '/' + pat)
    if i_vocab is None:
        counter = get_counter_dir_parallel(txt_dir, pat)
        vocab = get_vocab(counter, vocab_size)
    else:
        vocab = i_vocab
    print("... Vocabulary constructed ...")
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
                if len(v) > 5:
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
                fp.write(' 0\n')
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



