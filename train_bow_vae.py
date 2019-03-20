# coding: utf-8

import os, sys
import argparse


parser = argparse.ArgumentParser(description='Train a bag-of-words representation topic model as Variational AutoEncoder')
parser.add_argument('--train_dir', type=str, help='Directory containing files representing the input TRAINING data')
parser.add_argument('--test_dir', type=str, help='Directory containing files representing the input TEST data')
parser.add_argument('--file_pat', type=str, help='Regexp file pattern to match for documents (for TRAINING and TESTING data directories), [default = \'*\']', default='*')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--seed', type=int, default=1234, help='The random seed to use for RNG')
parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (against test data) during training')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.005)
parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=200)
parser.add_argument('--n_latent', type=int, help='Number of latent dimensions (topics)', default=20)
parser.add_argument('--latent_distribution', type=str, help='Latent distribution (logistic_gaussian|vmf|gaussian|gaussian_unitvar)',
                    default='vmf')
parser.add_argument('--kappa', type=float, default=100.0, help='Kappa value for vMF latent distribution')
parser.add_argument('--tr_vec_file', type=str, help='Training file in sparse vector format')
parser.add_argument('--tst_vec_file', type=str, help='Test/validation file in sparse vector format')
parser.add_argument('--vocab_file', type=str, help='Vocabulary file associated with sparse vector data')
parser.add_argument('--max_vocab_size', type=int, help='Maximum vocabulary size', default=2000)
parser.add_argument('--target_sparsity', type=float,
                    help='Target weight decoder sparsity. Default (0.0) means no sparsity enforced', default=0.0)
parser.add_argument('--sparsity_threshold', type=float, default=1e-3,
                    help='Threshold under which a weight is deemed close to zero for estimating sparsity')
parser.add_argument('--init_sparsity_pen', type=float, default = 0.0)
parser.add_argument('--hidden_dim', type=int, help='Dimension of hidden layers in encoder and network (default 200)', default=200)
parser.add_argument('--save_dir', type=str, default='_experiments')
parser.add_argument('--trace_file', type=str, default=None, help='Trace: (epoch, perplexity, NPMI) into a separate file for producing training curves')
parser.add_argument('--model_dir', type=str, default=None, help='Save final model and associated meta-data to this directory (default None)')
parser.add_argument('--hybridize', action='store_true', help='Use Symbolic computation graph (i.e. MXNet hybridize)')
parser.add_argument('--coherence_regularizer_penalty', type=float, help='Use word-embedding coherence regularization', default=0.0)
parser.add_argument('--embedding_source', type=str, help='Use pre-trained embedding to initialize first encoder layer', default=None)
parser.add_argument('--embedding_size', type=int, help='Embedding size when no pre-trained embedding is used (default 300)', default=300)
parser.add_argument('--fixed_embedding', action='store_true', help='Keep embedding layer fixed during training', default=False)
parser.add_argument('--use_labels_as_covars', action='store_true', help='If labels/meta-data are provided, use as covariates in model', default=False)
parser.add_argument('--topic_seed_file', type=str, default=None, help='Seed topic terms')

args = parser.parse_args()

from tmnt.bow_runner import train_bow_vae

train_bow_vae(args)

