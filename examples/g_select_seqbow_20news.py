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


data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)

# %%
# Example TODO ...




