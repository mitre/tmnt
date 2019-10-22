"""
Copyright (c) 2019 The MITRE Corporation.
"""

import argparse

def get_base_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_vec_file', type=str, help='Training file in sparse vector format')
    parser.add_argument('--tst_vec_file', type=str, help='Test/validation file in sparse vector format')
    parser.add_argument('--heldout_vec_file', type=str, help='Helout test file in sparse vector format')
    parser.add_argument('--vocab_file', type=str, help='Vocabulary file associated with sparse vector data')
    parser.add_argument('--seed', type=int, default=1234, help='The random seed to use for RNG')
    parser.add_argument('--save_dir', type=str, default='_experiments')
    parser.add_argument('--model_dir', type=str, default=None, help='Save final model and associated meta-data to this directory (default None)')
    parser.add_argument('--use_labels_as_covars', action='store_true', help='If labels/meta-data are provided, use as covariates in model', default=False)
    parser.add_argument('--scalar_covars', action='store_true', help='If labels/meta-data are provided, treat value as scalar rather than categorical',
                        default=False)
    parser.add_argument('--topic_seed_file', type=str, default=None, help='Seed topic terms')
    parser.add_argument('--num_final_evals', type=int, help='Number of times to evaluate selected configuration (with random initializations)', default=1)
    parser.add_argument('--init_sparsity_pen', type=float, default = 0.00001)
    parser.add_argument('--sparsity_threshold', type=float, default = 0.001)
    parser.add_argument('--str_encoding', type=str, default='utf-8')
    parser.add_argument('--hybridize', action='store_true', help='Use Symbolic computation graph (i.e. MXNet hybridize)')
    parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)    
    return parser

