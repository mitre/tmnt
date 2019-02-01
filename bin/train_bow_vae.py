# coding: utf-8

import os, sys
import argparse


parser = argparse.ArgumentParser(description='Train a bag-of-words representation topic model as Variational AutoEncoder')
parser.add_argument('--train_dir', type=str, help='Directory containing files representing the input TRAINING data')
parser.add_argument('--file_pat', type=str, help='Regexp file pattern to match for documents')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=16)
parser.add_argument('--n_latent', type=int, help='Number of latent dimensions (topics)', default=64)
parser.add_argument('--tr_vec_file', type=str, help='Training file in sparse vector format')
parser.add_argument('--tst_vec_file', type=str, help='Test/validation file in sparse vector format')
parser.add_argument('--vocab_file', type=str, help='Vocabulary file associated with sparse vector data')

args = parser.parse_args()

LIB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LIB_ROOT, "tmnt"))
from bow_runner import train_bow_vae

train_bow_vae(args)

