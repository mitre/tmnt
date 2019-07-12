# coding: utf-8

import io
import json
import gluonnlp as nlp
import glob
from gluonnlp.data import Counter
from multiprocessing import Pool, cpu_count
import threading

from tmnt.preprocess import BasicTokenizer

__all__ = ['JsonVectorizer', 'TextVectorizer']


class Vectorizer(object):

    def __init__(self, custom_stop_word_file=None):
        self.tokenizer = BasicTokenizer(use_stop_words=True)

    def get_counter_dir_parallel(self, data_dir, pat):
        raise NotImplementedError('Vectorizer must be instantiated as TextVectorizer or JsonVectorizer')

    def vectorize_fn(self, file_and_vocab):
        raise NotImplementedError('Vectorizer fn must be specified by concrete subclass')
    
    def get_vocab(self, counter, size):
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=5, max_size=size)
        return vocab

    def get_sparse_vecs(self, sp_out_file, vocab_out_file, data_dir, vocab_size=2000, i_vocab=None, full_histogram_file=None, pat='*.json'):
        files = glob.glob(data_dir + '/' + pat)
        if i_vocab is None:
            counter = self.get_counter_dir_parallel(data_dir, pat)
            vocab = self.get_vocab(counter, vocab_size)
        else:
            vocab = i_vocab
        files_and_vocab = [(f,vocab) for f in files]
        if len(files) > 2:
            p = Pool(cpu_count())
            sp_vecs = p.map(self.vectorize_fn, files_and_vocab)
        else:
            sp_vecs = map(self.vectorize_fn, files_and_vocab)
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


class JsonVectorizer(Vectorizer):

    def __init__(self, custom_stop_word_file=None, text_key='body', min_doc_size=6):
        super(JsonVectorizer, self).__init__(custom_stop_word_file)
        self.text_key = text_key
        self.min_doc_size = min_doc_size
    
    def get_counter_file(self, json_file):
        counter = None
        with io.open(json_file, 'r') as fp:
            for l in fp:
                js = json.loads(l)
                txt = js[self.text_key] ## text field
                counter = nlp.data.count_tokens(self.tokenizer.tokenize(txt), counter = counter)
        return counter

    def get_counter_dir_parallel(self, data_dir, pat):
        files = glob.glob(data_dir + '/' + pat)
        if len(files) > 2:
            p = Pool(cpu_count())
            counters = p.map(self.get_counter_file, files)
        else:
            counters = map(self.get_counter_file, files)
        return sum(counters, Counter())

    def vectorize_fn(self, file_and_vocab):
        json_file, vocab = file_and_vocab
        sp_vecs = []
        with io.open(json_file, 'r') as fp:
            for l in fp:
                js = json.loads(l)
                toks = self.tokenizer.tokenize(js[self.text_key])
                tok_ids = [vocab[token] for token in toks if token in vocab]
                if (len(tok_ids) >= self.min_doc_size):
                    cnts = nlp.data.count_tokens(tok_ids)
                    sp_vecs.append(sorted(cnts.items()))
        return sp_vecs


class TextVectorizer(Vectorizer):

    def __init__(self, custom_stop_word_file=None, min_doc_size=6):
        super(TextVectorizer, self).__init__(custom_stop_word_file)
        self.min_doc_size = min_doc_size


    def get_counter_dir_parallel(self, txt_dir, pat='*.txt'):
        def batches(l, n):
            for i in range(0, len(l), n):
                yield l[i:i+n]
        files = glob.glob(txt_dir + '/' + pat)
        batch_size = max(1, int(len(files) / 20))
        file_batches = list(batches(files, batch_size))
        if len(file_batches) > 2:
            p = Pool(cpu_count())
            counters = p.map(self.get_counter_file_batch, file_batches)        
        else:
            counters = map(self.get_counter_file_batch, file_batches)    
        return sum(counters, Counter())


    def get_counter_file_batch(self, txt_file_batch):
        counter = Counter()
        for txt_file in txt_file_batch:
            with io.open(txt_file, 'r') as fp:
                for txt in fp:
                    counter = nlp.data.count_tokens(self.tokenizer.tokenize(txt), counter = counter)
        return counter


    def vectorize_fn(self, txt_file_and_vocab):
        txt_file, vocab = txt_file_and_vocab
        sp_vecs = []
        with io.open(txt_file, 'r') as fp:
            doc_tok_ids = []
            for txt in fp:
                toks = self.tokenizer.tokenize(txt)
                tok_ids = [vocab[token] for token in toks if token in vocab]
                doc_tok_ids.extend(tok_ids)
            cnts = nlp.data.count_tokens(doc_tok_ids)
            sp_vecs.append(sorted(cnts.items()))
        return sp_vecs

    
