# coding: utf-8

import os, sys
import argparse

from tmnt.seq_vae.train_berttrans import train_main

parser = argparse.ArgumentParser(description='Train a Transformer-based Variational AutoEncoder on Context-aware Encodings')

parser.add_argument('--input_file', type=str, help='Directory containing a RecordIO file representing the input data')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='bertadam')
parser.add_argument('--gen_lr', type=float, help='General learning rate', default=0.00001)
parser.add_argument('--gpus',type=str, help='GPU device ids', default='')
parser.add_argument('--save_dir',type=str, help='Target directory for trained model parameters', default='cvae_model_out')
parser.add_argument('--batch_size',type=int, help='Training batch size', default=8)
parser.add_argument('--num_filters',type=int, help='Number of filters in first layer (each subsequent layer uses x2 filters)', default=64)
parser.add_argument('--latent_dim',type=int, help='Encoder dimensionality', default=256)
parser.add_argument('--wd_embed_dim',type=int, help='Word embedding dimensionality', default=256)
parser.add_argument('--dec_layers',type=int, help='Decoder transformer layers', default=6)
parser.add_argument('--kld_wt',type=float, help='Weight of the KL divergence term in variational loss', default=1.0)
parser.add_argument('--sent_size',type=int, help='Fixed/max length of sentence (zero padded); should be power of 2', default=16)
parser.add_argument('--save_model_freq', type=int, help='Number of epochs to save intermediate model', default=100)
parser.add_argument('--weight_decay', type=float, default=0.00001)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--offset_factor', type=float, default=1.0)
parser.add_argument('--min_lr', type=float, default=1e-7)
parser.add_argument('--wd_temp', type=float, help='Temperature coefficient for output embedding' ,default=0.01)


args = parser.parse_args()

train_main(args)

