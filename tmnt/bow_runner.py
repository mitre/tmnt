# coding: utf-8

import math

import mxnet as mx
from mxnet import autograd
from mxnet import gluon

from tmnt.bow_vae.bow_doc_loader import DataIterLoader, collect_sparse_data, BowDataSet, collect_stream_as_sparse_matrix
from tmnt.bow_vae.bow_models import BowNTM

l1_dim = 500

def train(args, vocabulary, data_train_csr, total_tr_words, data_test_csr=None, total_tst_words=0, ctx=mx.cpu()):
    train_iter = mx.io.NDArrayIter(data_train_csr, None, args.batch_size, last_batch_handle='discard', shuffle=True)
    train_dataloader = DataIterLoader(train_iter)
    if data_test_csr is not None:
        test_iter = mx.io.NDArrayIter(data_test_csr, None, args.batch_size, last_batch_handle='discard', shuffle=False)
        test_dataloader = DataIterLoader(test_iter)
    model = BowNTM(args.batch_size, len(vocabulary), l1_dim, args.n_latent)
    model.l1_pen_const.initialize()
    model.encoder.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=False)
    model.generator.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=False)
    model.decoder.initialize(mx.init.Uniform(1.0), ctx=ctx, force_reinit=False)

    model.hybridize(static_alloc=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    new_l1_coef = args.init_sparsity_pen
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
        ## Update L1 coefficient
        if args.target_sparsity > 0.0:            
            dec_weights = model.decoder.collect_params().get('weight').data().abs()
            ratio_small_weights = (dec_weights < args.sparsity_threshold).sum().asscalar() / dec_weights.size
            new_l1_coef = new_l1_coef * math.pow(2.0, args.target_sparsity - ratio_small_weights)
            print("Setting L1 coeffficient to {} [sparsity ratio = {}]".format(new_l1_coef, ratio_small_weights))
            model.l1_pen_const.set_data(mx.nd.array([new_l1_coef]))
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
