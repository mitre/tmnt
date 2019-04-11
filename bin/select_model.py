# coding: utf-8

import os, sys
import argparse

from tmnt.bow_vae.train import model_select_bow_vae
from tmnt.common_params import get_base_argparser

parser = get_base_argparser()
parser.description = 'Automated model selection for TMNT Topic Models'
parser.add_argument('--config_space', type=str, help='YAML configuration file that specifies the configuration space for model selection')

args = parser.parse_args()

os.environ["MXNET_STORAGE_FALLBACK_LOG_VERBOSE"] = "0"

model_select_bow_vae(args)

