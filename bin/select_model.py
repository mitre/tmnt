# coding: utf-8

import os, sys
import argparse

from tmnt.bow_vae.train import model_select_bow_vae
from tmnt.common_params import get_base_argparser

parser = get_base_argparser()
parser.description = 'Automated model selection for TMNT Topic Models'
parser.add_argument('--config_space', type=str, help='YAML configuration file that specifies the configuration space for model selection')
parser.add_argument('--budget',type=int, help='Maximum number of training epochs in model search')
parser.add_argument('--iterations',type=int, help='Maximum number of full model training epochs to carry out as part of search', default=4)
parser.add_argument('--coherence_coefficient', type=float, help='Weight applied to coherence (NPMI) term of objective function', default=1.0)
parser.add_argument('--ns_port', type=int, help='Force a specific port number for HPBandSter nameserver', default=None)

args = parser.parse_args()

os.environ["MXNET_STORAGE_FALLBACK_LOG_VERBOSE"] = "0"

model_select_bow_vae(args)

