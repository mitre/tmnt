# coding: utf-8

import os, sys
import argparse


parser = argparse.ArgumentParser(description='Train a bag-of-words representation topic model as Variational AutoEncoder')
parser.add_argument('--train_dir', type=str, help='Directory containing files representing the input TRAINING data')
parser.add_argument('--test_dir', type=str, help='Directory containing files representing the input TEST data')
parser.add_argument('--file_pat', type=str, help='Regexp file pattern to match for documents (for TRAINING and TESTING data directories), [default = \'*\']', default='*')
parser.add_argument('--seed', type=int, default=1234, help='The random seed to use for RNG')
parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (against test data) during training')
parser.add_argument('--model_select', action='store_true', help='Use model selection (Experimental)')
parser.add_argument('--config_space', type=str, help='YAML configuration file that specifies the configuration space for model selection')
parser.add_argument('--config_instance', type=str, help='Configuration instance specifying all hyperparameters for a single training run')
parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=200)
parser.add_argument('--tr_vec_file', type=str, help='Training file in sparse vector format')
parser.add_argument('--tst_vec_file', type=str, help='Test/validation file in sparse vector format')
parser.add_argument('--vocab_file', type=str, help='Vocabulary file associated with sparse vector data')
parser.add_argument('--max_vocab_size', type=int, help='Maximum vocabulary size', default=2000)
parser.add_argument('--save_dir', type=str, default='_experiments')
parser.add_argument('--trace_file', type=str, default=None, help='Trace: (epoch, perplexity, NPMI) into a separate file for producing training curves')
parser.add_argument('--model_dir', type=str, default=None, help='Save final model and associated meta-data to this directory (default None)')
parser.add_argument('--hybridize', action='store_true', help='Use Symbolic computation graph (i.e. MXNet hybridize)')
parser.add_argument('--use_labels_as_covars', action='store_true', help='If labels/meta-data are provided, use as covariates in model', default=False)
parser.add_argument('--topic_seed_file', type=str, default=None, help='Seed topic terms')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit -- also the maximum BUDGET')


args = parser.parse_args()

os.environ["MXNET_STORAGE_FALLBACK_LOG_VERBOSE"] = "0"

from tmnt.bow_runner import train_bow_vae

train_bow_vae(args)

