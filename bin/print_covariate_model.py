# coding: utf-8

import os, io
import argparse
import funcy
import json
from pathlib import Path

from tmnt.bow_vae.runtime import BowNTMInference

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=Path, help='Directory with trained model files')
parser.add_argument('--vec_file', type=str, help='Sparse vector file')
parser.add_argument('--num_terms', type=int, help='Number of terms to display for each topic (for each covariate value)')
parser.add_argument('--covariate_values', type=str, help='File with covariate values in ordered list')
parser.add_argument('--output_file', type=str, help='Text file output')

args = parser.parse_args()
param_file, config_file, vocab_file = \
    args.model_dir / "model.params", args.model_dir / "model.config", args.model_dir / "vocab.json"

infer = BowNTMInference(param_file, config_file, vocab_file)

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
        
    
