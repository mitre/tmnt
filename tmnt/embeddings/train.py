# coding: utf-8

import argparse
import logging
import os
import random
import sys
import time
import io
import datetime

from data import transform_data_fasttext, transform_data_word2vec, preprocess_dataset_stream, preprocess_dataset, CustomDataSet
from embed_utils import logging_config

import mxnet as mx
import numpy as np

import gluonnlp as nlp
from model import SG, CBOW

def get_context(args):
    if args.gpu is None or args.gpu == '':
        context = [mx.cpu()]
    elif isinstance(args.gpu, int):
        context = [mx.gpu(args.gpu)]
    else:
        context = [mx.gpu(int(i)) for i in args.gpu]
    return context



def train_embeddings(args, exp_folder):
    """Training helper."""
    if not args.model.lower() in ['cbow', 'skipgram']:
        logging.error('Unsupported model %s.', args.model)
        sys.exit(1)

    if args.data_type == 'custom':
        data = CustomDataSet(args.data_root,args.file_pattern, '<bos>', '<eos>', skip_empty=True)
        data, vocab, idx_to_counts = preprocess_dataset_stream(data, logging, max_vocab_size = args.max_vocab_size)
    elif args.data_type == 'gbw':
        data = nlp.data.corpora.google_billion_word.GBWStream()
        data, vocab, idx_to_counts = preprocess_dataset_stream(data, logging, max_vocab_size = args.max_vocab_size)
    elif args.data_type == 'text8':
        data = nlp.data.Text8(segment='train')
        data, vocab, idx_to_counts = preprocess_dataset(
            data, max_vocab_size=args.max_vocab_size)

    logging.info('Data pre-processing complete.  Data transform beginning...')

    if args.ngram_buckets > 0:
        data, batchify_fn, subword_function = transform_data_fasttext(
            data, vocab, idx_to_counts, cbow=args.model.lower() == 'cbow',
            ngram_buckets=args.ngram_buckets, ngrams=args.ngrams,
            batch_size=args.batch_size, window_size=args.window,
            frequent_token_subsampling=args.frequent_token_subsampling)
    else:
        subword_function = None
        data, batchify_fn = transform_data_word2vec(
            data, vocab, idx_to_counts, cbow=args.model.lower() == 'cbow',
            batch_size=args.batch_size, window_size=args.window,
            frequent_token_subsampling=args.frequent_token_subsampling)

    num_tokens = float(sum(idx_to_counts))

    model = CBOW if args.model.lower() == 'cbow' else SG
    embedding = model(token_to_idx=vocab.token_to_idx, output_dim=args.emsize,
                      batch_size=args.batch_size, num_negatives=args.negative,
                      negatives_weights=mx.nd.array(idx_to_counts),
                      subword_function=subword_function)
    context = get_context(args)
    embedding.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=True, static_shape=True)

    optimizer_kwargs = dict(learning_rate=args.lr)
    try:
        trainer = mx.gluon.Trainer(embedding.collect_params(), args.optimizer,
                                   optimizer_kwargs)
    except ValueError as e:
        if args.optimizer == 'groupadagrad':
            logging.warning('MXNet <= v1.3 does not contain '
                            'GroupAdaGrad support. Falling back to AdaGrad')
            trainer = mx.gluon.Trainer(embedding.collect_params(), 'adagrad',
                                       optimizer_kwargs)
        else:
            raise e

    try:
        if args.no_prefetch_batch:
            data = data.transform(batchify_fn)
        else:
            from executors import LazyThreadPoolExecutor
            num_cpu = len(os.sched_getaffinity(0))
            ex = LazyThreadPoolExecutor(num_cpu)
    except (ImportError, SyntaxError, AttributeError):
        # Py2 - no async prefetching is supported
        logging.warning(
            'Asynchronous batch prefetching is not supported on Python 2. '
            'Consider upgrading to Python 3 for improved performance.')
        data = data.transform(batchify_fn)

    num_update = 0
    prefetched_iters = []
    for _ in range(min(args.num_prefetch_epoch, args.epochs)):
        prefetched_iters.append(iter(data))
    for epoch in range(args.epochs):
        if epoch + len(prefetched_iters) < args.epochs:
            prefetched_iters.append(iter(data))
        data_iter = prefetched_iters.pop(0)

        try:
            batches = ex.map(batchify_fn, data_iter)
        except NameError:  # Py 2 or batch prefetching disabled
            batches = data_iter

        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        for i, batch in enumerate(batches):
            ctx = context[i % len(context)]
            batch = [array.as_in_context(ctx) for array in batch]
            with mx.autograd.record():
                loss = embedding(*batch)
            loss.backward()

            num_update += loss.shape[0]
            if len(context) == 1 or (i + 1) % len(context) == 0:
                trainer.step(batch_size=1)

            # Logging
            log_wc += loss.shape[0]
            log_avg_loss += loss.mean().as_in_context(context[0])
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
                # Due to subsampling, the overall number of batches is an upper
                # bound
                num_batches = num_tokens // args.batch_size
                if args.model.lower() == 'skipgram':
                    num_batches = (num_tokens * args.window * 2) // args.batch_size
                else:
                    num_batches = num_tokens // args.batch_size
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'.format(
                                 epoch, i + 1, num_batches, log_avg_loss,
                                 wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

            #if args.eval_interval and (i + 1) % args.eval_interval == 0:
            #    with print_time('mx.nd.waitall()'):
            #        mx.nd.waitall()
            #    with print_time('evaluate'):
            #        evaluate(args, embedding, vocab, num_update)

    # Evaluate
    #with print_time('mx.nd.waitall()'):
    #    mx.nd.waitall()
    #with print_time('evaluate'):
    #    evaluate(args, embedding, vocab, num_update,
    #             eval_analogy=not args.no_eval_analogy)

    # Save full embedding
    #with print_time('save parameters'):
    ## write out vocab    
    with io.open(os.path.join(exp_folder, 'vocab.json'), 'w') as f:
        js_vocab = vocab.to_json()
        f.write(js_vocab)
    embedding.embedding.save_parameters(os.path.join(exp_folder, 'embedding.params'))
    for t in ['vector', 'emperor', 'boy', 'queen', 'trump', 'clinton', 'senator', 'president']:
        ls = get_k_closest_tokens(vocab, embedding.embedding, 10, t)
        logging.info('K closest tokens to {} are {}'.format(t, ls))


def norm_vecs_by_row(x):
    return x / (mx.nd.sum(x * x, axis=1) + 1e-10).sqrt().reshape((-1, 1))


def get_k_closest_tokens(vocab, embedding, k, word):
    word_vec = norm_vecs_by_row(embedding[[word]])    
    vocab_vecs = norm_vecs_by_row(embedding[vocab._idx_to_token])
    dot_prod = mx.nd.dot(vocab_vecs, word_vec.T)
    indices = mx.nd.topk(
        dot_prod.reshape((len(vocab._idx_to_token), )),
        k=k + 1,
        ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    result = [vocab.idx_to_token[i] for i in indices[1:]]
    return result



def train(args):
    i_dt = datetime.datetime.now()
    exp_folder = '{}/exp_{}_{}_{}_{}_{}_{}'.format(args.logdir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
    logging_config(exp_folder)
    logging.info(args)
    train_embeddings(args, exp_folder)