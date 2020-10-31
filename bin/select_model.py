# coding: utf-8

import os, sys
import argparse

from tmnt.selector import model_select_bow_vae
from tmnt.common_params import get_base_argparser

parser = get_base_argparser()
parser.description = 'Automated model selection for TMNT Topic Models'
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
    model_select_bow_vae(args)

