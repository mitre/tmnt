# coding: utf-8

import os, sys
import argparse

from tmnt.selector import model_select_seq_bow
from tmnt.common_params import get_base_argparser

parser = argparse.ArgumentParser(description='Model selection for Transformer VED Topic Model')
parser.add_argument('--tr_file', type=str, help='A JSON list file representing the training data')
parser.add_argument('--val_file', type=str, help='A JSON list file representing the validation data (optional)')
parser.add_argument('--tst_file', type=str, help='A JSON list file representing the test data (optional)')
parser.add_argument('--use_gpu',action='store_true', help='Use GPU(s) if available', default=False)
parser.add_argument('--model_dir', type=str, help='Directory for final saved model files', default=None)
parser.add_argument('--save_dir',type=str, help='Target directory for trained model parameters', default='seqvae_exp_logs')
parser.add_argument('--bow_vocab_file', type=str, help='Vocabulary for BOW in decoder', default=None)
parser.add_argument('--sent_size',type=int, help='Fixed/max length of sentence (zero padded); should be power of 2', default=32)
parser.add_argument('--weight_decay', type=float, help='Learning weight decay', default=0.00001)
parser.add_argument('--offset_factor', type=float, help='Adjusts offset for LR decay; values < 1 are faster', default=1.0)
parser.add_argument('--log_interval', type=int, help='Number of batches after which loss and reconstruction examples will be logged', default=20)
parser.add_argument('--num_final_evals', type=int, help='Number of times to evaluate selected configuration (with random initializations)', default=1)
parser.add_argument('--seed', type=int, default=1234, help='The random seed to use for RNG')
parser.add_argument('--config_space', type=str, help='YAML configuration file that specifies the configuration space for model selection')
parser.add_argument('--iterations',type=int, help='Maximum number of full model training epochs to carry out as part of search', default=16)
parser.add_argument('--coherence_coefficient', type=float, help='Weight applied to coherence (NPMI) term of objective function', default=1.0)
parser.add_argument('--brackets', type=int, help='Number of hyberband brackets', default=1)
parser.add_argument('--searcher', type=str, help='Autogluon search method (random, bayesopt)', default='random')
parser.add_argument('--scheduler', type=str, default='hyperband', help='Scheduler: (hyperband or fifo)')
parser.add_argument('--cpus_per_task', type=int, help='Number of CPUs to allocate for each evaluation (set higher to decrease parallelism)', default = 2)


args = parser.parse_args()


if __name__ == '__main__':
    os.environ["MXNET_STORAGE_FALLBACK_LOG_VERBOSE"] = "0"
    model_select_seq_bow(args)

