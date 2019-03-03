# coding: utf-8

import math
import logging
import datetime
import io
import os
import json
import mxnet as mx
import numpy as np
from mxnet import autograd
from mxnet import gluon
import gluonnlp as nlp

from tmnt.bow_vae.bow_doc_loader import DataIterLoader, collect_sparse_data, BowDataSet, collect_stream_as_sparse_matrix
from tmnt.bow_vae.bow_models import BowNTM, MetaDataBowNTM
from tmnt.utils.log_utils import logging_config
from tmnt.coherence.npmi import EvaluateNPMI


def get_wd_freqs(data_csr, max_sample_size=10000):
    sample_size = min(max_sample_size, data_csr.shape[0])
    data = data_csr[:sample_size].asnumpy() # only take first 10000 to estimate frequencies - but should select at RANDOM
    sums = np.sum(data, axis=0)
    return list(sums)


def train(args, vocabulary, data_train_csr, total_tr_words, data_test_csr=None, total_tst_words=0, train_labels=None, test_labels=None, ctx=mx.cpu()):
    wd_freqs = get_wd_freqs(data_train_csr)
    emb_size = vocabulary.embedding.idx_to_vec[0].size if vocabulary.embedding else args.embedding_size
    if args.use_labels_as_covars and train_labels is not None:
        n_covars = mx.nd.max(train_labels).asscalar() + 1
        train_labels = mx.nd.one_hot(train_labels, n_covars)
        test_labels = mx.nd.one_hot(test_labels, n_covars) if test_labels is not None else None
        model = \
            MetaDataBowNTM(n_covars,vocabulary, args.hidden_dim, args.n_latent, emb_size,
                           fixed_embedding=args.fixed_embedding, latent_distrib=args.latent_distribution,
                           coherence_reg_penalty=args.coherence_regularizer_penalty,
                           batch_size=args.batch_size, wd_freqs=wd_freqs, ctx=ctx)
    else:
        model = \
            BowNTM(vocabulary, args.hidden_dim, args.n_latent, emb_size, fixed_embedding=args.fixed_embedding, latent_distrib=args.latent_distribution,
                   coherence_reg_penalty=args.coherence_regularizer_penalty,
                   batch_size=args.batch_size, wd_freqs=wd_freqs, ctx=ctx)
    train_iter = mx.io.NDArrayIter(data_train_csr, train_labels, args.batch_size, last_batch_handle='discard', shuffle=True)
    train_dataloader = DataIterLoader(train_iter)    
    if data_test_csr is not None:
        test_iter = mx.io.NDArrayIter(data_test_csr, test_labels, args.batch_size, last_batch_handle='discard', shuffle=False)
        test_dataloader = DataIterLoader(test_iter)        
    if (args.hybridize):
        model.hybridize(static_alloc=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    l1_coef = args.init_sparsity_pen
    for epoch in range(args.epochs):
        epoch_loss = 0
        total_rec_loss = 0
        total_l1_pen = 0
        tr_size = 0
        for i, (data, labels) in enumerate(train_dataloader):
            tr_size += data.shape[0]
            if labels is None or labels.size == 0:
                labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
                labels = labels.as_in_context(ctx)
            data = data.as_in_context(ctx)
            with autograd.record():
                elbo, rec_loss, l1_pen, _ = model(data, labels) if args.use_labels_as_covars else model(data)
                elbo_mean = elbo.mean()
            elbo_mean.backward()
            trainer.step(data.shape[0]) ## step based on batch size
            total_l1_pen += l1_pen.sum().asscalar()
            total_rec_loss += rec_loss.sum().asscalar()
            epoch_loss += elbo_mean.asscalar()            
        perplexity = math.exp(total_rec_loss / total_tr_words)
        logging.info("Epoch {}: Loss = {}, Training perplexity = {} [ L1 Pen = {} ] [ Rec loss = {}]".
                     format(epoch, epoch_loss / tr_size, perplexity, total_l1_pen/tr_size, total_rec_loss/tr_size))
        if args.target_sparsity > 0.0:            
            dec_weights = model.decoder.collect_params().get('weight').data().abs()
            ratio_small_weights = (dec_weights < args.sparsity_threshold).sum().asscalar() / dec_weights.size
            l1_coef = l1_coef * math.pow(2.0, args.target_sparsity - ratio_small_weights)
            logging.info("Setting L1 coeffficient to {} [sparsity ratio = {}]".format(l1_coef, ratio_small_weights))
            model.l1_pen_const.set_data(mx.nd.array([l1_coef]))
        if (epoch + 1) % args.eval_freq == 0:
            evaluate(model, test_dataloader, total_tst_words, args, ctx)
    log_top_k_words_per_topic(model, vocabulary, args.n_latent, 10)
    coherence_file = args.tst_vec_file if args.tst_vec_file else args.tr_vec_file
    log_coherence(model, vocabulary, args.n_latent, 10, data_test_csr)
    return model


def evaluate(model, data_loader, total_words, args, ctx=mx.cpu(), debug=False):
    total_rec_loss = 0
    for i, (data,labels) in enumerate(data_loader):
        if labels is None:            
            labels = mx.nd.expand_dims(mx.nd.zeros(data.shape[0]), 1)
            labels = labels.as_in_context(ctx)
        data = data.as_in_context(ctx)
        _, rec_loss, _, log_out = model(data, labels) if args.use_labels_as_covars else model(data)
        total_rec_loss += rec_loss.sum().asscalar()
    perplexity = math.exp(total_rec_loss / total_words)
    logging.info("TEST/VALIDATION Perplexity = {}".format(perplexity))
    return perplexity


def log_coherence(model, vocab, num_topics, k, test_data):
    w = model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    num_topics = min(num_topics, sorted_ids.shape[-1])
    top_k_words_per_topic = [[int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]
    npmi_eval = EvaluateNPMI(top_k_words_per_topic)
    #npmi = npmi_eval.evaluate_sp_vec(test_file)
    npmi = npmi_eval.evaluate_csr_mat(test_data)
    logging.info("Test Coherence: {}".format(npmi))


def log_top_k_words_per_topic(model, vocab, num_topics, k):
    w = model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    for t in range(num_topics):
        top_k = [ vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
        term_str = ' '.join(top_k)
        logging.info("Topic {}: {}".format(str(t), term_str))


def train_bow_vae(args):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
    logging_config(folder=train_out_dir, name='bow_ntm', level=logging.INFO)
    tr_file = args.train_dir
    if args.vocab_file and args.tr_vec_file:
        vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words, tr_labels, tst_labels = \
            collect_sparse_data(args.tr_vec_file, args.vocab_file, args.tst_vec_file)
    else:
        tr_dataset = BowDataSet(args.train_dir, args.file_pat)    
        tr_csr_mat, vocab, total_tr_words = collect_stream_as_sparse_matrix(tr_dataset, max_vocab_size=args.max_vocab_size)
        tr_labels = None
        tst_labels = None
        if args.test_dir:
            tst_dataset = BowDataSet(args.test_dir, args.file_pat)
            tst_csr_mat, _, total_tst_words = collect_stream_as_sparse_matrix(tst_dataset, pre_vocab=vocab)
    ctx = mx.cpu() if args.gpu is None or args.gpu == '' or int(args.gpu) < 0 else mx.gpu(int(args.gpu))
    if args.embedding_source:
        glove_twitter = nlp.embedding.create('glove', source=args.embedding_source)
        vocab.set_embedding(glove_twitter)
        emb_size = len(vocab.embedding.idx_to_vec[0])
        for word in vocab.embedding._idx_to_token:
            if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                vocab.embedding[word] = mx.nd.random.normal(-1.0, 1.0, emb_size)
        
    ### XXX - NOTE: For smaller datasets, may make sense to convert sparse matrices to dense here up front
    m = train(args, vocab, tr_csr_mat, total_tr_words, tst_csr_mat, total_tst_words, tr_labels, tst_labels, ctx=ctx)
    if args.model_dir:
        pfile = os.path.join(args.model_dir, 'model.params')
        sp_file = os.path.join(args.model_dir, 'model.specs')
        vocab_file = os.path.join(args.model_dir, 'vocab.json')
        m.save_parameters(pfile)
        sp = {}
        sp['enc_dim'] = args.hidden_dim
        sp['n_latent'] = args.n_latent
        sp['latent_distribution'] = args.latent_distribution
        sp['emb_size'] = vocab.embedding.idx_to_vec[0].size if vocab.embedding else args.embedding_size
        specs = json.dumps(sp)
        with open(sp_file, 'w') as f:
            f.write(specs)
        with open(vocab_file, 'w') as f:
            f.write(vocab.to_json())
