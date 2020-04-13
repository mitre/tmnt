# codeing: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import math
import os
import numpy as np
import logging
import json
import datetime
import io
import gluonnlp as nlp
import string
import re
import mxnet as mx
import mxnet.ndarray as F
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from tmnt.coherence.npmi import EvaluateNPMI
from tmnt.bow_vae.bow_doc_loader import load_vocab
from tmnt.seq_bow_ved.models import TransformerBowVED, TransformerBowVEDTest, BertBowVED
from tmnt.utils.log_utils import logging_config
from tmnt.seq_bow_ved.sb_data_loader import load_dataset_basic_seq_bow, load_dataset_bert

def get_wd_freqs(data_csr, max_sample_size=1000000):
    sample_size = min(max_sample_size, data_csr.shape[0])
    data = data_csr[:sample_size] 
    sums = mx.nd.sum(data, axis=0)
    return sums


def get_basic_model(args, bow_vocab_size, vocab, emb_dim, wd_freqs, ctx):
    model = TransformerBowVED(bow_vocab_size, vocab, emb_dim, args.latent_dist, num_units=args.num_units,
                              hidden_size=args.hidden_size,
                              num_heads=args.num_heads,
                              n_latent=args.latent_dim, max_sent_len=args.sent_size,
                              transformer_layers=args.transformer_layers,
                              kappa = args.kappa, 
                              batch_size=args.batch_size,
                              wd_freqs=wd_freqs,
                              kld=args.kld_wt, ctx=ctx)
    return model


def get_bert_model(args, bert_base, bow_vocab_size, wd_freqs, ctx):
    model = BertBowVED(bert_base, bow_vocab_size, args.latent_dist, 
                              n_latent=args.latent_dim, max_sent_len=args.sent_size,
                              kappa = args.kappa, 
                              batch_size=args.batch_size,
                              kld=args.kld_wt, wd_freqs=wd_freqs, ctx=ctx)
    return model


def compute_coherence(model, bow_vocab, k, test_data, log_terms=False, ctx=mx.cpu()):
    num_topics = model.n_latent
    sorted_ids = model.get_top_k_terms(k)
    num_topics = min(num_topics, sorted_ids.shape[-1])
    top_k_words_per_topic = [[ int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]
    npmi_eval = EvaluateNPMI(top_k_words_per_topic)
    npmi = npmi_eval.evaluate_csr_mat(test_data)
    logging.info("Test Coherence: {}".format(npmi))
    if log_terms:
        top_k_tokens = [list(map(lambda x: bow_vocab.idx_to_token[x], list(li))) for li in top_k_words_per_topic]
        for i in range(num_topics):
            logging.info("Topic {}: {}".format(i, top_k_tokens[i]))
    return npmi


def train_bow_seq_ved(args, model, bow_vocab, data_train, train_csr, data_test=None, ctx=mx.cpu(), use_bert=False):
    
    dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size,
                                           shuffle=True, last_batch='rollover')
    if data_test:
        dataloader_test = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size,
                                               shuffle=False) if data_test else None

    num_train_examples = len(data_train)
    num_train_steps = int(num_train_examples / args.batch_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    differentiable_params = []
    
    lr = args.gen_lr

    
    gen_trainer = gluon.Trainer(model.encoder.collect_params(), args.optimizer,
                            {'learning_rate': args.gen_lr, 'epsilon': 1e-6, 'wd':args.weight_decay})
    lat_trainer = gluon.Trainer(mode.latent_dist.collect_params(), 'adam', {'learning_rate': args.gen_lr, 'epsilon': 1e-6})
    dec_trainer = gluon.Trainer(mode.decoder.collect_params(), 'adam', {'learning_rate': args.gen_lr, 'epsilon': 1e-6})    

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in model.encoder.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)
    
    for epoch_id in range(args.epochs):
        step_loss = 0
        step_recon_ls = 0
        step_kl_ls = 0
        for batch_id, seqs in enumerate(dataloader):
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / ((num_train_steps - num_warmup_steps) * args.offset_factor)
                new_lr = max(lr - offset, args.min_lr)
            gen_trainer.set_learning_rate(new_lr)
            with mx.autograd.record():
                if use_bert:
                    input_ids, valid_length, type_ids, output_vocab = seqs
                    ls, recon_ls, kl_ls, predictions = model(input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                                                             valid_length.astype('float32').as_in_context(ctx),
                                                             output_vocab.as_in_context(ctx))
                else:
                    input_ids, output_vocab = seqs
                    ls, recon_ls, kl_ls, predictions = model(input_ids.as_in_context(ctx), output_vocab.as_in_context(ctx))
                ls = ls.mean()
            ls.backward()
            grads = [p.grad(ctx) for p in differentiable_params]
            gluon.utils.clip_global_norm(grads, 1)
            gen_trainer.step(1) # step of 1 since we averaged loss over batch
            lat_trainer.step(1)
            dec_trainer.step(1) # update decoder trainer associated weights
            step_loss += ls.asscalar()
            step_recon_ls += recon_ls.mean().asscalar()
            step_kl_ls += kl_ls.mean().asscalar()
            if (batch_id + 1) % (args.log_interval) == 0:
                logging.info('[Epoch {}/{} Batch {}/{}] loss={:.4f}, recon_loss={:.4f}, kl_loss={:.4f}, gen_lr={:.7f}'
                             .format(epoch_id, args.epochs, batch_id + 1, len(dataloader),
                                     step_loss / args.log_interval, step_recon_ls / args.log_interval, step_kl_ls / args.log_interval,
                                     gen_trainer.learning_rate))
                step_loss = 0
                step_recon_ls = 0
                step_kl_ls = 0
                _ = compute_coherence(model, bow_vocab, 10, train_csr, log_terms=True)
        if (epoch_id + 1) % args.save_interval == 0:
            write_model(model, args, epoch_id)
    write_model(model, args)


def write_model(m, args, epoch_id=0):
    if args.model_dir:
        suf = '_'+ str(epoch_id) if epoch_id > 0 else ''
        pfile = os.path.join(args.model_dir, ('model.params' + suf))
        conf_file = os.path.join(args.model_dir, ('model.config' + suf))
        vocab_file = os.path.join(args.model_dir, ('vocab.json' + suf))
        m.save_parameters(pfile)
        dd = {}
        dd['latent_dist'] = m.latent_distrib
        dd['num_units'] = m.num_units
        dd['num_heads'] = m.num_heads        
        dd['hidden_size'] = m.hidden_size
        dd['n_latent'] = m.n_latent
        dd['transformer_layers'] = m.transformer_layers
        dd['kappa'] = m.kappa
        dd['sent_size'] = m.max_sent_len
        dd['embedding_size'] = m.wd_embed_dim
        specs = json.dumps(dd)
        with open(conf_file, 'w') as f:
            f.write(specs)
        with open(vocab_file, 'w') as f:
            f.write(m.vocabulary.to_json())


def train_main(args):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
    print("Set logging config to {}".format(train_out_dir))
    logging_config(folder=train_out_dir, name='train_trans_vae', level=logging.INFO, no_console=False)
    logging.info(args)
    context = mx.cpu() if args.gpus is None or args.gpus == '' else mx.gpu(int(args.gpus))
    bow_vocab = load_vocab(args.bow_vocab_file)
    
    if args.use_bert:
        data_train, bert_base, vocab, data_csr = load_dataset_bert(args.input_file, len(bow_vocab), max_len=args.sent_size, ctx=context)
        wd_freqs = get_wd_freqs(data_csr)
        model = get_bert_model(args, bert_base, len(bow_vocab), wd_freqs, context)
        pad_id = vocab[vocab.padding_token]
        train_bow_seq_ved(args, model, bow_vocab, data_train, data_csr, data_test=None, ctx=context, use_bert=True)
    else:
        data_train, vocab, data_csr, _ = load_dataset_basic_seq_bow(args.input_file, len(bow_vocab),
                                                       vocab=None, json_text_key=args.json_text_key, max_len=args.sent_size,
                                                                    max_vocab_size=args.max_vocab_size, ctx=context)
        emb = None
        if args.embedding_source:
            emb = nlp.embedding.create('glove', source = args.embedding_source)
        if emb:
            vocab.set_embedding(emb)
            _, emb_size = vocab.embedding.idx_to_vec.shape
            oov_items = 0
            for word in vocab.embedding._idx_to_token:
                if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                    oov_items += 1
                    vocab.embedding[word] = mx.nd.random.normal(0.0, 0.1, emb_size)
            logging.info("** There are {} out of vocab items **".format(oov_items))
        else:
            logging.info("** No pre-trained embedding provided, learning embedding weights from scratch **")
        wd_freqs = get_wd_freqs(data_csr)
        if vocab.embedding is not None:
            emb_dim = len(vocab.embedding.idx_to_vec[0])
        else:
            emb_dim = args.wd_embed_dim

        model = get_basic_model(args, len(bow_vocab), vocab, emb_dim, wd_freqs, context)
        pad_id = vocab[vocab.padding_token]
        train_bow_seq_ved(args, model, bow_vocab, data_train, data_csr, data_test=None, ctx=context, use_bert=False)
            

