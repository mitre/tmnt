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


parser = argparse.ArgumentParser(description='Train a Transformer-based Variational AutoEncoder on Context-aware Encodings')

parser.add_argument('--input_file', type=str, help='Directory containing a RecordIO file representing the input data')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--bert_lr',type=float, help='BERT Learning rate', default=0.00001)
parser.add_argument('--gen_lr', type=float, help='General (nonBERT) learning rate', default=0.0001)
parser.add_argument('--gpus',type=str, help='GPU device ids', default='')
parser.add_argument('--save_dir',type=str, help='Target directory for trained model parameters', default='cvae_model_out')
parser.add_argument('--batch_size',type=int, help='Training batch size', default=8)
parser.add_argument('--num_filters',type=int, help='Number of filters in first layer (each subsequent layer uses x2 filters)', default=64)
parser.add_argument('--latent_dim',type=int, help='Encoder dimensionality', default=256)
parser.add_argument('--wd_embed_dim',type=int, help='Word embedding dimensionality', default=256)
parser.add_argument('--kld_wt',type=float, help='Weight of the KL divergence term in variational loss', default=1.0)
parser.add_argument('--sent_size',type=int, help='Fixed/max length of sentence (zero padded); should be power of 2', default=16)
parser.add_argument('--batch_report_freq', type=int, help='Frequency to report batch stats during training', default=10)
parser.add_argument('--save_model_freq', type=int, help='Number of epochs to save intermediate model', default=100)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--bert_warmup_ratio', type=float, default=0.1)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--offset_factor', type=float, default=1.0)
parser.add_argument('--min_lr', type=float, default=1e-7)


args = parser.parse_args()
i_dt = datetime.datetime.now()
train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
print("Set logging config to {}".format(train_out_dir))
logging_config(folder=train_out_dir, name='train_cvae', level=logging.INFO, no_console=False)
logging.info(args)

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
    #model.out_embedding.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
    model.inv_embed.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
    
    #model.hybridize(static_alloc=True)

    bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size,
                                           shuffle=True, last_batch='rollover')
    #bert_dataloader_test = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size,
    #                                           shuffle=False) if data_test else None

    num_train_examples = len(data_train)
    num_train_steps = int(num_train_examples / args.batch_size * args.epochs)
    warmup_ratio = args.bert_warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    differentiable_params = []


    #bert_trainer = gluon.Trainer(model.bert.collect_params(), args.optimizer,
    #                        {'learning_rate': args.bert_lr, 'epsilon': 1e-9, 'wd':args.weight_decay})

    non_bert_params = gluon.parameter.ParameterDict()
    for prs in [model.mu_encoder.collect_params(), model.lv_encoder.collect_params(),
                model.decoder.collect_params(), model.out_embedding.collect_params()]:
        non_bert_params.update(prs)
    #gen_optimizer = mx.optimizer.Adam(learning_rate=args.gen_lr,
    #                                  lr_scheduler=CosineAnnealingSchedule(args.min_lr, args.gen_lr, num_train_steps))
    decayed_updates = int(num_train_steps * 0.8)
    gen_optimizer = mx.optimizer.Adam(learning_rate=args.gen_lr,
                                  clip_gradient=5.0,
                                  lr_scheduler=mx.lr_scheduler.CosineScheduler(decayed_updates,
                                                                               args.gen_lr,
                                                                               args.min_lr,
                                                                               warmup_steps=int(decayed_updates/10),
                                                                               warmup_begin_lr=(args.gen_lr / 10),
                                                                               warmup_mode='linear'
                                                                               ))

    #gen_trainer = gluon.Trainer(non_bert_params, gen_optimizer)
    gen_trainer = gluon.Trainer(model.collect_params(), gen_optimizer)

    lr = args.bert_lr

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    ## change to only do this for BERT parameters - will clip the gradients
    #for p in model.bert.collect_params().values():
    #    if p.grad_req != 'null':
    #        differentiable_params.append(p)
        
    for epoch_id in range(args.epochs):
        step_loss = 0
        for batch_id, seqs in enumerate(bert_dataloader):
            step_num += 1
            #if step_num < num_warmup_steps:
            #    new_lr = lr * step_num / num_warmup_steps
            #else:
            #    offset = (step_num - num_warmup_steps) * lr / ((num_train_steps - num_warmup_steps) * args.offset_factor)
            #    new_lr = max(lr - offset, args.min_lr)
            #bert_trainer.set_learning_rate(new_lr)
            with mx.autograd.record():
                input_ids, valid_length, type_ids = seqs
                ls, predictions = model(input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                                valid_length.astype('float32').as_in_context(ctx))
            ls.backward()
            #grads = [p.grad(ctx) for p in differentiable_params]
            #gluon.utils.clip_global_norm(grads, 1)
            #bert_trainer.step(1)  # BERT param updates not adjusted by batch size ...
            gen_trainer.step(input_ids.shape[0], ignore_stale_grad=True) # let rest of model be updated by batch size
            step_loss += ls.mean().asscalar()
            if (batch_id + 1) % (args.log_interval) == 0:
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, bert_lr={:.7f}, gen_lr={:.7f}'
                             .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                     step_loss / args.log_interval,
                                     gen_trainer.learning_rate, gen_trainer.learning_rate))
                step_loss = 0
            if (batch_id + 1) % args.log_interval == 0:
                if report_fn:
                    mx.nd.waitall()
                    report_fn(input_ids, predictions)


class CosineAnnealingSchedule():
    def __init__(self, min_lr, max_lr, cycle_length):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        
    def __call__(self, iteration):
        if iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
            return adjusted_cycle
        else:
            return self.min_lr



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
        

if __name__ == '__main__':
    context = mx.cpu() if args.gpus is None or args.gpus == '' else mx.gpu(int(args.gpus))
    data_train, bert_base, vocab = load_dataset(args.input_file, max_len=args.sent_size, ctx=context)
    report_fn = get_report_reconstruct_data_fn(vocab)
    train_berttrans_vae(data_train, bert_base, context, report_fn)
        
