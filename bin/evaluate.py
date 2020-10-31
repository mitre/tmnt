#!/usr/bin/env python3

import math
import argparse
import io
import os
import mxnet as mx

from tmnt.inference import BowVAEInferencer
from tmnt.data_loading import file_to_data, load_vocab
from tmnt.eval_npmi import NPMI, EvaluateNPMI
from tmnt.utils.ngram_helpers import BigramReader
import gluonnlp as nlp

from itertools import combinations

import umap
from pathlib import Path


import numpy as np

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Variational AutoEncoder topic model')
    parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
    parser.add_argument('--test_file', type=str, required=True, help='file in sparse vector format')    
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file associated with sparse vector data')
    parser.add_argument('--model_dir', type=Path,
                        help='The directory where the params, specs, and vocab should be found.')
    parser.add_argument('--plot_file', type=str, help='Output plot')
    parser.add_argument('--words_per_topic', type=int, help='Number of terms per topic to output', default=10)
    parser.add_argument('--override_top_k_terms', type=str, help='File of topic terms to use instead of those from model', default=None)
    return parser

def read_vector_file(file):
    labels = []
    docs = []
    with open(file) as f:
        for line in map(str.strip, f):
            label, *words = line.split()
            labels.append(int(label))
            docs.append(list(map(lambda t: int(t.split(":")[0]), words)))
    return labels, docs

def evaluate(inference, data_loader, total_words, ctx=mx.cpu()):
    total_rec_loss = 0
    for i, (data,_) in enumerate(data_loader):
        data = data.as_in_context(ctx)
        _, rec_loss, _, log_out = inference.model(data)
        total_rec_loss += rec_loss.sum().asscalar()
    perplexity = math.exp(total_rec_loss / total_words)
    return perplexity


def get_top_k_terms_from_file(in_file):
    top_k_terms = []
    with io.open(in_file, 'r') as fp:
        for l in fp:
            ts = [t.strip() for t in l.split(',')]
            top_k_terms.append(ts)
    return top_k_terms


os.environ["MXNET_STORAGE_FALLBACK_LOG_VERBOSE"] = "0"

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    verbose = False ### XXX - add as argument
    vocab = load_vocab(args.vocab_file)        
    if args.override_top_k_terms:
        top_k_words_per_topic = get_top_k_terms_from_file(args.override_top_k_terms)
        tst_csr, _, _, _ = file_to_data(args.test_file, len(vocab))
        top_k_words_per_topic_ids = [ [ vocab[t] for t in t_set ]  for t_set in top_k_words_per_topic ]
        npmi_eval = EvaluateNPMI(top_k_words_per_topic_ids)
        test_npmi = npmi_eval.evaluate_csr_mat(tst_csr)
        print("**** Test NPMI = {} *******".format(test_npmi))
        exit(0)

    param_file, config_file, vocab_file = \
        args.model_dir / "model.params", args.model_dir / "model.config", args.model_dir / "vocab.json"
    
    inference_model = BowVAEInferencer(param_file, config_file, vocab_file,
                                      ctx=mx.cpu() if args.gpu < 0 else mx.gpu(args.gpu))


    if args.plot_file: # get UMAP embedding visualization
        import matplotlib.pyplot as plt
        encoded, labels = inference_model.encode_vec_file(args.test_file)
        encodings = np.array([doc.asnumpy() for doc in encoded])
        print("There are {0} labels and {1} encodings".format(len(labels), len(encodings)))
        umap_model = umap.UMAP(n_neighbors=4, min_dist=0.5, metric='euclidean')
        embeddings = umap_model.fit_transform(encodings)
        plt.scatter(*embeddings.T, c=labels, s=0.2, alpha=0.7, cmap='coolwarm')
        plt.savefig(args.plot_file, dpi=1000)

    top_k_words_per_topic = inference_model.get_top_k_words_per_topic(args.words_per_topic)        
    for i in range(len(top_k_words_per_topic)):
        print("Topic {}: {}".format(i, top_k_words_per_topic[i]))

    top_k_words_per_topic_ids = [ [ inference_model.vocab[t] for t in t_set ]  for t_set in top_k_words_per_topic ]

    npmi_eval = EvaluateNPMI(top_k_words_per_topic_ids)
    tst_csr, _, _, _ = file_to_data(args.test_file, len(vocab))
    test_npmi = npmi_eval.evaluate_csr_mat(tst_csr)
    print("**** Test NPMI = {} *******".format(test_npmi))
    exit(0)



