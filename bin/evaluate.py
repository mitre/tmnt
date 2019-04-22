#!/usr/bin/env python3

import math
import argparse

import mxnet as mx

from tmnt.bow_vae.runtime import BowNTMInference
from tmnt.bow_vae.bow_doc_loader import collect_sparse_data
from tmnt.coherence.npmi import NPMI, EvaluateNPMI
from tmnt.utils.ngram_helpers import BigramReader

from itertools import combinations

import umap
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Variational AutoEncoder topic model')
    parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
    parser.add_argument('--train_file', type=str, required=True, help='file in sparse vector format')        
    parser.add_argument('--test_file', type=str, required=True, help='file in sparse vector format')    
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file associated with sparse vector data')
    parser.add_argument('--model_dir', required=True, type=Path,
                        help='The directory where the params, specs, and vocab should be found.')
    parser.add_argument('--plot_file', type=str, help='Output plot')
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



if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    verbose = False ### XXX - add as argument

    param_file, config_file, vocab_file = \
        args.model_dir / "model.params", args.model_dir / "model.config", args.model_dir / "vocab.json"

    inference_model = BowNTMInference(param_file, config_file, vocab_file,
                                      ctx=mx.cpu() if args.gpu < 0 else mx.gpu(args.gpu))



    if args.plot_file: # get UMAP embedding visualization
        encoded, labels = inference_model.encode_vec_file(args.test_file)
        encodings = np.array([doc.asnumpy() for doc in encoded])
        print("There are {0} labels and {1} encodings".format(len(labels), len(encodings)))
        umap_model = umap.UMAP(n_neighbors=5, min_dist=0.1, metric='euclidean')
        embeddings = umap_model.fit_transform(encodings)
        plt.scatter(*embeddings.T, c=labels, s=0.2, alpha=0.7, cmap='coolwarm')
        plt.savefig(args.plot_file, dpi=1000)

    top_k_words_per_topic = inference_model.get_top_k_words_per_topic(10)        
    for i in range(len(top_k_words_per_topic)):
        print("Topic {}: {}".format(i, top_k_words_per_topic[i]))

    npmi_eval = EvaluateNPMI(top_k_words_per_topic)
    _, tr_csr, _, tst_csr, _, _, _ = collect_sparse_data(args.train_file, args.vocab_file, args.test_file)
    print("Shape of test_csr = {}".format(tst_csr.shape))
    test_npmi = npmi_eval.evaluate_csr_mat(tst_csr)
    print("**** Test NPMI = {} *******".format(test_npmi))
              


