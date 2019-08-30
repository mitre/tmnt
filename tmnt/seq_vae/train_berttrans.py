# codeing: utf-8

import argparse, tarfile
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
from tmnt.seq_vae.trans_seq_models import BertTransVAE
from tmnt.utils.log_utils import logging_config
from tmnt.seq_vae.tokenization import FullTokenizer, EncoderTransform


trans_table = str.maketrans(dict.fromkeys(string.punctuation))

def remove_punct_and_urls(txt):
    string = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '', txt) ## wipe out URLs
    return string.translate(trans_table)


def load_dataset(sent_file, max_len=64, ctx=mx.cpu()):
    train_arr = []
    with io.open(sent_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            if len(line.split(' ')) > 4:
                train_arr.append(line)
    bert_model = 'bert_12_768_12'
    dname = 'book_corpus_wiki_en_uncased'
    bert_base, vocab = nlp.model.get_model(bert_model,  
                                             dataset_name=dname,
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
    tokenizer = FullTokenizer(vocab, do_lower_case=True)
    transformer = EncoderTransform(tokenizer, max_len, clean_fn=remove_punct_and_urls)
    data_train = gluon.data.SimpleDataset(train_arr).transform(transformer)
    return data_train, bert_base, vocab


def train_berttrans_vae(data_train, bert_base, ctx=mx.cpu(), report_fn=None):
    model = BertTransVAE(bert_base, wd_embed_dim=args.wd_embed_dim, n_latent=args.latent_dim, max_sent_len=args.sent_size, batch_size=args.batch_size,
                       kld=args.kld_wt, ctx=ctx)
    model.mu_encoder.initialize(init=mx.init.Normal(0.1), ctx=ctx)
    model.lv_encoder.initialize(init=mx.init.Normal(0.1), ctx=ctx)    
    model.decoder.initialize(init=mx.init.Xavier(magnitude=2.34), ctx=ctx)
    model.out_embedding.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
    model.inv_embed.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
    
    #model.hybridize(static_alloc=True)

    bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size,
                                           shuffle=True, last_batch='rollover')
    #bert_dataloader_test = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size,
    #                                           shuffle=False) if data_test else None

    num_train_examples = len(data_train)
    num_train_steps = int(num_train_examples / args.batch_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    differentiable_params = []


    gen_trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                            {'learning_rate': args.gen_lr, 'epsilon': 1e-6, 'wd':args.weight_decay})

    #bert_trainer = gluon.Trainer(model.bert.collect_params(), args.optimizer,
    #                        {'learning_rate': args.gen_lr, 'epsilon': 1e-6, 'wd':args.weight_decay})

    #non_bert_params = gluon.parameter.ParameterDict()
    #for prs in [model.mu_encoder.collect_params(), # model.lv_encoder.collect_params(),
    #            model.decoder.collect_params(), model.out_embedding.collect_params(), model.inv_embed.collect_params()]:
    #    non_bert_params.update(prs)

    #non_bert_optimizer = mx.optimizer.Adam(learning_rate=args.gen_lr,
    #                                  lr_scheduler=CosineAnnealingSchedule(args.min_lr, args.gen_lr, num_train_steps))
    #gen_optimizer = mx.optimizer.Adam(learning_rate=args.gen_lr,
    #                                  lr_scheduler=CosineAnnealingSchedule(args.min_lr, args.gen_lr, num_train_steps))
    #decayed_updates = int(num_train_steps * 0.8)
    #gen_optimizer = mx.optimizer.Adam(learning_rate=args.gen_lr,
    #                              clip_gradient=5.0,
    #                              lr_scheduler=mx.lr_scheduler.CosineScheduler(decayed_updates,
    #                                                                           args.gen_lr,
    #                                                                           args.min_lr,
    #                                                                           warmup_steps=int(decayed_updates/10),
    #                                                                           warmup_begin_lr=(args.gen_lr / 10),
    #                                                                           warmup_mode='linear'
    #                                                                           ))

    #gen_trainer = gluon.Trainer(non_bert_params, gen_optimizer)
    #gen_trainer = gluon.Trainer(model.collect_params(), gen_optimizer)



    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    ## change to only do this for BERT parameters - will clip the gradients
    for p in model.bert.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    lr = args.gen_lr
    
    for epoch_id in range(args.epochs):
        step_loss = 0
        ntmp = model.inv_embed.set_temp(epoch_id, args.epochs) # adjust temp parameter based on current epoch
        logging.info(">>> Setting Inverse Embedding temp to {}".format(ntmp))
        for batch_id, seqs in enumerate(bert_dataloader):
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / ((num_train_steps - num_warmup_steps) * args.offset_factor)
                new_lr = max(lr - offset, args.min_lr)
            gen_trainer.set_learning_rate(new_lr)
            with mx.autograd.record():
                input_ids, valid_length, type_ids = seqs
                ls, predictions = model(input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                                valid_length.astype('float32').as_in_context(ctx))
                ls = ls.mean()
            ls.backward()
            #grads = [p.grad(ctx) for p in differentiable_params]
            #gluon.utils.clip_global_norm(grads, 1)
            gen_trainer.step(1) # step of 1 since we averaged loss over batch
            step_loss += ls.asscalar()
            if (batch_id + 1) % (args.log_interval) == 0:
                logging.info('[Epoch {}/{} Batch {}/{}] loss={:.4f}, gen_lr={:.7f}'
                             .format(epoch_id, args.epochs, batch_id + 1, len(bert_dataloader),
                                     step_loss / args.log_interval,
                                     gen_trainer.learning_rate, gen_trainer.learning_rate))
                step_loss = 0
            if (batch_id + 1) % args.log_interval == 0:
                if report_fn:
                    mx.nd.waitall()
                    report_fn(input_ids, predictions)


def get_report_reconstruct_data_fn(vocab, pad_id=0):
    def report_reconstruct_data_fn(data, predictions):
        reconstructed_sent_ids = mx.nd.argmax(predictions[0],1).asnumpy() ## get the first item of batch and arg_max over vocab size
        input_sent_ids = data[0].asnumpy()
        rec_sent = [vocab.idx_to_token[int(i)] for i in reconstructed_sent_ids if i != pad_id]   # remove <PAD> token from rendering
        in_sent = [vocab.idx_to_token[int(i)] for i in input_sent_ids if i != pad_id]
        in_ids = [str(i) for i in input_sent_ids]
        logging.info("---------- Reconstruction Output/Comparison --------")
        logging.info("Input Ids = {}".format(' '.join(in_ids)))
        logging.info("Input = {}".format(' '.join(in_sent)))
        logging.info("Reconstructed = {}".format(' '.join(rec_sent)))
    return report_reconstruct_data_fn
        

def train_main(args):
    i_dt = datetime.datetime.now()
    train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
    print("Set logging config to {}".format(train_out_dir))
    logging_config(folder=train_out_dir, name='train_cvae', level=logging.INFO, no_console=False)
    logging.info(args)
    context = mx.cpu() if args.gpus is None or args.gpus == '' else mx.gpu(int(args.gpus))
    data_train, bert_base, vocab = load_dataset(args.input_file, max_len=args.sent_size, ctx=context)
    report_fn = get_report_reconstruct_data_fn(vocab)
    train_berttrans_vae(data_train, bert_base, context, report_fn)
        
