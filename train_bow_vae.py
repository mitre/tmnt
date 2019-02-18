# coding: utf-8

import os, sys
import argparse


parser = argparse.ArgumentParser(description='Train a bag-of-words representation topic model as Variational AutoEncoder')
parser.add_argument('--train_dir', type=str, help='Directory containing files representing the input TRAINING data')
parser.add_argument('--file_pat', type=str, help='Regexp file pattern to match for documents')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.0001)
parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=32)
parser.add_argument('--n_latent', type=int, help='Number of latent dimensions (topics)', default=32)
parser.add_argument('--latent_distribution', type=str, help='Latent distribution (logistic_gaussian|vmf|gaussian)',
                    default='logistic_gaussian')
parser.add_argument('--kappa', type=float, default=100.0, help='Kappa value for vMF latent distribution')
parser.add_argument('--tr_vec_file', type=str, help='Training file in sparse vector format')
parser.add_argument('--tst_vec_file', type=str, help='Test/validation file in sparse vector format')
parser.add_argument('--vocab_file', type=str, help='Vocabulary file associated with sparse vector data')
parser.add_argument('--target_sparsity', type=float,
                    help='Target weight decoder sparsity. Default (0.0) means no sparsity enforced', default=0.0)
parser.add_argument('--sparsity_threshold', type=float, default=1e-3,
                    help='Threshold under which a weight is deemed close to zero for estimating sparsity')
parser.add_argument('--init_sparsity_pen', type=float, default = 0.0)
parser.add_argument('--hidden_dim', type=int, help='Dimension of hidden layers in encoder and network', default=300)
parser.add_argument('--num_gen_layers', type=int, help='Number of fully connected layers in generator', default=0)
parser.add_argument('--save_dir', type=str, default='_experiments')
parser.add_argument('--model_dir', type=str, default=None, help='Save final model and associated meta-data to this directory (default None)')
parser.add_argument('--hybridize', action='store_true', help='Use Symbolic computation graph (i.e. MXNet hybridize)')
parser.add_argument('--coherence_regularizer_penalty', type=float, help='Use word-embedding coherence regularization', default=0.0)

args = parser.parse_args()

from tmnt.bow_runner import train_bow_vae

train_bow_vae(args)

