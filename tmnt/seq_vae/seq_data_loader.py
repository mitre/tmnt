import io
import itertools
import os
import logging

import gluonnlp as nlp
import mxnet as mx
from mxnet.gluon import nn
from mxnet import autograd as ag
import string
from tmnt.seq_vae.tokenization import FullTokenizer, EncoderTransform, BasicTokenizer

trans_table = str.maketrans(dict.fromkeys(string.punctuation))

def remove_punct_and_urls(txt):
    string = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '', txt) ## wipe out URLs
    return string.translate(trans_table)


def load_dataset_bert(sent_file, max_len=64, ctx=mx.cpu()):
    train_arr = []
    with io.open(sent_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            if len(line.split(' ')) > 4:
                train_arr.append(line)
    bert_model = 'bert_12_768_12'
    dname = 'book_corpus_wiki_en_uncased'
    bert_base, vocab = nlp.model.get_model(bert_model,  
                                             dataset_name=dname,
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
    tokenizer = FullTokenizer(vocab, do_lower_case=True)
    transformer = EncoderTransform(tokenizer, max_len, clean_fn=remove_punct_and_urls)
    data_train = gluon.data.SimpleDataset(train_arr).transform(transformer)
    return data_train, bert_base, vocab


def load_dataset_basic(sent_file, vocab=None, max_len=64, ctx=mx.cpu()):
    train_arr = []
    tokenizer = BasicTokenizer(do_lower_case=True)
    if not vocab:        
        counter = None
        with io.open(sent_file, 'r', encoding='utf-8') as fp:
            for line in fp:
                if len(line.split(' ')) > 4:
                    toks = tokenizer.tokenize(line)[:(max_len-2)]
                    counter = nlp.data.count_tokens(toks, counter = counter)
        vocab = nlp.Vocab(counter)
    pad_id = vocab[vocab.padding_token]
    with io.open(sent_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            if len(line.split(' ')) > 4:
                toks = tokenizer.tokenize(line)[:(max_len-2)]
                toks = ['<bos>'] + toks + ['<eos>']
                ids = [vocab[t] for t in toks]
                padded_ids = ids[:max_len] if len(ids) >= max_len else ids + pad_id * (max_len - len(ids))
                train_arr.append(padded_ids)
    data_train = gluon.data.SimpleDataset(train_arr)
    return data_train, vocab




