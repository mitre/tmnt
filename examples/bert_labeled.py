from tmnt.estimator import FullyLabeledSeqEstimator
import numpy as np
import gluonnlp as nlp
import os
import mxnet as mx
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.bert_handling import get_bert_datasets


tr_file = '/Users/wellner/Projects/MILU/ARGMINE/json/ART_sent_val.json'
dev_file = '/Users/wellner/Projects/MILU/ARGMINE/json/ART_sent_val.json'
model_name = 'bert_12_768_12'
dataset = 'book_corpus_wiki_en_uncased'
class_labels = ["Objectives", "Outcome", "Prior", "Approach"]
batch_size = 32
seq_len = 128
pad = False

tr_dataset, dev_dataset, num_examples, bert_base = get_bert_datasets(class_labels,
                                                                     tr_file, dev_file, model_name,
                                                                     dataset, batch_size, 8, seq_len, pad, mx.cpu())

print("num_examples = {}".format(num_examples))

estimator = FullyLabeledSeqEstimator(bert_base, len(class_labels), ctx=mx.cpu(), log_interval=1)

estimator.fit_with_validation(tr_dataset, dev_dataset, num_examples)

