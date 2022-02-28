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
from tmnt.selector import BaseSelector
from tmnt.configuration import TMNTConfigSeqBOW, default_seq_config_space
from tmnt.trainer import SeqBowVEDTrainer
import autogluon.core as ag


data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)

tr_size = 8000
train_data = data[:tr_size]
dev_data   = data[-tr_size:]
train_y    = y[:tr_size]
dev_y      = y[-tr_size:]
model_name = 'bert_12_768_12'
dataset = 'book_corpus_wiki_en_uncased'
batch_size = 32
seq_len = 128
pad = True

use_logging = True

if use_logging:    
    logging_config(folder='.', name='f_seqbow_20news', level='info', console_level='info')
    log_method = 'log'
else:
    log_method = 'print'

train_y_s = ['class_'+str(y) for y in train_y]
dev_y_s = ['class_'+str(y) for y in dev_y]

## Or - if training a pure unsupervised topic model:
#train_y_s = [ None for _ in train_y ]
#dev_y_s   = [ None for _ in dev_y ]

#config_space = './ved_config.yaml'
#tmnt_config = TMNTConfigSeqBOW(config_space).get_configspace()
tmnt_config = default_seq_config_space
selector = BaseSelector(tmnt_config, iterations=2, searcher='random', 
                        scheduler='fifo', cpus_per_task=8, log_dir='_exp_out_dir')

trainer = SeqBowVEDTrainer('_exp_out_dir', (train_data, train_y_s), (dev_data, dev_y_s), use_gpu=False, log_interval=1)
estimator, _, _ = selector.select_model(trainer)

## to explicity/separately validate the trained estimator

from tmnt.bert_handling import get_bert_datasets

class_labels = list(set(train_y_s)) # get unique labels here
dataloader = get_bert_tokenized_dataset(train_data, None, class_labels, max_len=tmnt_config['max_seq_len'])
result_dict, _ , _ = estimator.validate(estimator.model, dataloader)





