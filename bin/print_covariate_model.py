# coding: utf-8

import os, io
import argparse
import funcy
import json
import numpy as np
from pathlib import Path

from tmnt.bow_vae.runtime import BowNTMInference
from tmnt.bow_vae.sensitivity_analysis import get_jacobians_at_data_file, get_encoder_jacobians_at_data_file

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=Path, help='Directory with trained model files')
parser.add_argument('--vec_file', type=str, help='Sparse vector file')
parser.add_argument('--num_terms', type=int, help='Number of terms to display for each topic (for each covariate value)')
parser.add_argument('--covariate_values', type=str, help='File with covariate values in ordered list')
parser.add_argument('--scalar_covar_range', type=str, help='min/max comma-separated (default [0.0, 1.0])', default="0.0,1.0")
parser.add_argument('--num_covars', type=int, help='Number of points within interval', default=0)
parser.add_argument('--output_file', type=str, help='Text file output')

args = parser.parse_args()
param_file, config_file, vocab_file = \
    args.model_dir / "model.params", args.model_dir / "model.config", args.model_dir / "vocab.json"

infer = BowNTMInference(param_file, config_file, vocab_file)

def print_categorical_covar_model():    
    covars = []
    with open(args.covariate_values, 'r') as fp:
        for c in fp:
            covars.append(str.strip(c))
    topk_per_covar = infer.get_top_k_words_per_topic_per_covariate(args.num_terms)
    with open(args.output_file, 'w') as out:
        for c in covars:
            out.write("===== Topics for {} =====\n".format(c))
            ind = infer.label_map[c]
            c_topics = topk_per_covar[ind]
            i = 0
            for topic in c_topics:
                i += 1
                out.write("Topic {}: ".format(str(i)))
                for term in topic:
                    out.write("{} ".format(term))
                out.write('\n')

def _old_print_scalar_covar_model():
    min_v, max_v = args.scalar_covar_range.split(',')
    k = args.num_terms
    vocab = infer.model.vocabulary
    with open(args.output_file, 'w') as out:
        for c in np.linspace(float(min_v), float(max_v), num = args.num_covars):
            out.write("===== Topics for {} =====\n".format(c))
            sorted_ids = infer.model.get_top_k_terms_with_covar(10, c)
            n_topics = sorted_ids.shape[1]
            for t in range(n_topics):
                top_k = [ vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
                out.write("Topic {}".format(str(t)))
                for term in top_k:
                    out.write("{} ".format(term))
                out.write('\n')

def print_scalar_covar_model():
    min_v, max_v = args.scalar_covar_range.split(',')
    k = args.num_terms
    vocab = infer.model.vocabulary
    covars = np.linspace(float(min_v), float(max_v), num = args.num_covars)
    jacobians = get_jacobians_at_data(infer.model, args.vec_file, covars, sample_size=50)
    with open(args.output_file, 'w') as out:
        for i in range(jacobians.shape[0]):
            j_for_cv = jacobians[i]
            sorted_j = j_for_cv.argsort(axis=0, is_ascend=False)
            n_topics = sorted_j.shape[1]
            print("Shape sorted_j = {}".format(sorted_j.shape))
            for t in range(n_topics):
                top_k = [ vocab.idx_to_token[int(i)] for i in list(sorted_j[:k, t].asnumpy()) ]
                out.write("Topic {}".format(str(t)))
                for term in top_k:
                    out.write("{} ".format(term))
                out.write('\n')


def print_encoder_term_sensitivities():
    k = args.num_terms
    vocab = infer.model.vocabulary
    jacobians = get_encoder_jacobians_at_data(infer.model, args.vec_file, sample_size=50)
    with open(args.output_file, 'w') as out:
        sorted_j = jacobians.argsort(axis=1, is_ascend=False)
        n_topics = sorted_j.shape[0]
        print("Shape sorted_j = {}".format(sorted_j.shape))
        for t in range(n_topics):
            top_k = [ vocab.idx_to_token[int(i)] for i in list(sorted_j[t, :k].asnumpy()) ]
            out.write("Topic {}: ".format(str(t)))
            for term in top_k:
                out.write("{} ".format(term))
            out.write('\n')


if __name__ == '__main__':
    if args.num_covars > 0:
        #print_scalar_covar_model()
        print_encoder_term_sensitivities()
    else:
        print_categorical_covar_model()

