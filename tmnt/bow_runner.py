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

def train(args, vocabulary, data_train_csr, total_tr_words, data_test_csr=None, total_tst_words=0, ctx=mx.cpu()):
    train_iter = mx.io.NDArrayIter(data_train_csr, None, args.batch_size, last_batch_handle='discard', shuffle=True)
    train_dataloader = DataIterLoader(train_iter)
    if data_test_csr is not None:
        test_iter = mx.io.NDArrayIter(data_test_csr, None, args.batch_size, last_batch_handle='discard', shuffle=False)
        test_dataloader = DataIterLoader(test_iter)
    model = BowNTM(args.batch_size, len(vocabulary), l1_dim, args.n_latent)
    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)
    model.hybridize(static_alloc=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    for epoch in range(args.epochs):
        epoch_loss = 0
        total_rec_loss = 0
        total_l1_pen = 0
        for i, (data,_) in enumerate(train_dataloader):
            data = data.as_in_context(ctx)
            with autograd.record():
                elbo, rec_loss, l1_pen = model(data)
            elbo.backward()
            trainer.step(data.shape[0]) ## step based on batch size
            total_l1_pen += l1_pen.sum().asscalar()
            total_rec_loss += rec_loss.sum().asscalar()
            epoch_loss += elbo.sum().asscalar()            
        perplexity = math.exp(total_rec_loss / total_tr_words)
        print("Loss = {}, Training perplexity = {} [ L1 Pen = {} ]".format(epoch_loss, perplexity, total_l1_pen))
        evaluate(model, test_dataloader, total_tst_words, ctx)


def evaluate(model, data_loader, total_words, ctx=mx.cpu()):
    total_rec_loss = 0
    for i, (data,_) in enumerate(data_loader):
        data = data.as_in_context(ctx)
        _, rec_loss, _ = model(data)
        total_rec_loss += rec_loss.sum().asscalar()
    perplexity = math.exp(total_rec_loss / total_words)
    print("TEST/VALIDATION Perplexity = {}".format(perplexity))
    return perplexity


def train_bow_vae(args):
    tr_file = args.train_dir
    if args.vocab_file and args.tr_vec_file:
        vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words = \
            collect_sparse_data(args.tr_vec_file, args.vocab_file, args.tst_vec_file)
    else:
        tr_dataset = BowDataSet(args.train_dir, args.file_pat)    
        tr_csr_mat, vocab, total_tr_words = collect_stream_as_sparse_matrix(tr_dataset, max_vocab_size=2000)
        tst_dataset = BowDataSet(args.test_dir, args.file_pat)
        tst_csr_mat, _, total_tst_words = collect_stream_as_sparse_matrix(tst_dataset, pre_vocab=vocab)
    train(args, vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words)
