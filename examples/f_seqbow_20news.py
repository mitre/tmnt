"""
Training a seq2bow encoder-decoder model
========================================
"""
from tmnt.estimator import SeqBowEstimator
import numpy as np
import gluonnlp as nlp
import os
import mxnet as mx
import logging
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.bert_handling import get_bert_datasets
from tmnt.utils.log_utils import logging_config
from mxnet.gluon.data import ArrayDataset

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)

tr_size = 100
train_data = data[:tr_size]
dev_data   = data[-tr_size:]
train_y    = y[:tr_size]
dev_y      = y[-tr_size:]
model_name = 'bert_12_768_12'
dataset = 'book_corpus_wiki_en_uncased'
batch_size = 32
seq_len = 64
pad = True

vectorizer = TMNTVectorizer(vocab_size=2000)
vectorizer.fit_transform(train_data)

ctx = mx.cpu() ## or mx.gpu(N) if using GPU device=N

supervised  = True
use_logging = True

if supervised:
    num_classes = int(np.max(y) + 1)
    classes = ['class_'+str(i) for i in range(num_classes)]
else:
    num_classes = 0
    classes = None

if use_logging:    
    logging_config(folder='.', name='f_seqbow_20news', level='info', console_level='info')
    log_method = 'log'
else:
    log_method = 'print'

train_y_s = ['class_'+str(y) for y in train_y]
dev_y_s = ['class_'+str(y) for y in dev_y]

tr_ds = ArrayDataset(train_data, train_y_s)
dev_ds = ArrayDataset(dev_data, dev_y_s)

print("Classes = {}".format(classes))

tr_dataset, dev_dataset, aux_dataset, num_examples, bert_base, bert_vocab = \
    get_bert_datasets(classes, vectorizer, 
                      tr_ds, dev_ds, batch_size, seq_len,
                      bert_model_name=model_name,
                      bert_dataset=dataset,
                      num_classes=num_classes, aux_ds = dev_ds,
                      ctx=ctx)


estimator = SeqBowEstimator(bert_base, bert_model_name = model_name, bert_data_name = dataset,
                            n_labels = num_classes,
                            bow_vocab = vectorizer.get_vocab(),
                            optimizer='bertadam',
                            batch_size=batch_size, ctx=ctx, log_interval=1,
                            log_method=log_method, gamma=1.0, 
                            lr=2e-5, decoder_lr=0.02, epochs=1)


# this will take quite some time without a GPU!
estimator.fit_with_validation(tr_dataset, dev_dataset, None, num_examples)
print("Estimator: {}".format(estimator.model))
estimator.write_model('./_model_dir')

