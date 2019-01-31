# coding: utf-8

import argparse
import math
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
from sklearn.metrics import precision_recall_curve, average_precision_score

from bow_vae.bow_doc_loader import *
from bow_vae.bow_models import BowNTM

l1_dim = 300

def train(args, vocabulary, data_train_csr, total_num_words, data_val=None, ctx=mx.cpu()):
    train_iter = mx.io.NDArrayIter(data_train_csr, None, args.batch_size, last_batch_handle='discard', shuffle=False)
    train_dataloader = DataIterLoader(train_iter)
    model = BowNTM(args.batch_size, len(vocabulary), l1_dim, args.n_latent)
    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    for epoch in range(args.epochs):
        epoch_loss = 0
        total_rec_loss = 0
        for i, (data,_) in enumerate(train_dataloader):
            data = data.as_in_context(ctx)
            with autograd.record():
                elbo, rec_loss = model(data)
            elbo.backward()
            trainer.step(data.shape[0]) ## step based on batch size
            total_rec_loss += rec_loss.sum().asscalar()
            epoch_loss += elbo.sum().asscalar()
        perplexity = math.exp(total_rec_loss / total_num_words)
        print("Loss = {}, Training perplexity = {}".format(epoch_loss, perplexity))
        #logging.info("Epoch loss = {}".format(epoch_loss))


def train_bow_vae(args):
    tr_file = args.train_dir
    tr_dataset = BowDataSet(args.train_dir, args.file_pat)    
    tr_csr_mat, vocab, total_num_words = collect_stream_as_sparse_matrix(tr_dataset, max_vocab_size=2000)
    train(args, vocab, tr_csr_mat, total_num_words)
