#!/usr/bin/env python3

import math
import argparse

import mxnet as mx

from tmnt.bow_vae.runtime import BowNTMInference
from tmnt.coherence.pmi import PMI
from tmnt.coherence.npmi import NPMI
from tmnt.utils.ngram_helpers import UnigramReader, BigramReader

from itertools import combinations

import umap
from pathlib import Path

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Variational AutoEncoder topic model')
    parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
    parser.add_argument('--train_file', type=str, required=True, help='file in sparse vector format')
    parser.add_argument('--eval_file', type=str, required=True, help='file in sparse vector format')    
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file associated with sparse vector data')
    parser.add_argument('--model_dir', required=True, type=Path,
                        help='The directory where the params, specs, and vocab should be found.')
    parser.add_argument('--num_topics', type=int, required=True, help='The number of topics')
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

def get_top_k_words_per_topic(inference, num_topics, k):
    w = inference.model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    return [[inference.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]

def get_top_k_word_idx_per_topic(inference, num_topics, k):
    w = inference.model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    num_topics = min(num_topics, sorted_ids.shape[-1])
    return [[int(i) for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]

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

    param_file, specs_file, vocab_file = \
        args.model_dir / "model.params", args.model_dir / "model.specs", args.model_dir / "vocab.json"

    inference_model = BowNTMInference(param_file, specs_file, vocab_file,
                                      ctx=mx.cpu() if args.gpu < 0 else mx.gpu(args.gpu))

    top_k_words_per_topic = get_top_k_word_idx_per_topic(inference_model, 100, args.num_topics)

    encoded, labels = inference_model.encode_vec_file(args.eval_file)
    encodings = np.array([doc.asnumpy() for doc in encoded])

    print("There are {0} labels and {1} encodings".format(len(labels), len(encodings)))

    if args.plot_file: # get UMAP embedding visualization
        umap_model = umap.UMAP(n_neighbors=5, min_dist=0.1, metric='euclidean')
        embeddings = umap_model.fit_transform(encodings)
        plt.ylim(top=8,bottom=-8)
        plt.xlim(left=-6,right=7)
        plt.scatter(*embeddings.T, c=labels, s=0.2, alpha=0.7, cmap='coolwarm')
        plt.savefig(args.plot_file, dpi=1000)

    bigram_reader = BigramReader(args.train_file)

    npmi = NPMI(bigram_reader.unigrams, bigram_reader.bigrams, bigram_reader.n_docs)

    total_npmi = 0
    for i, words_per_topic in enumerate(top_k_words_per_topic):
        total_topic_npmi = 0
        N = len(words_per_topic)
        for (w1, w2) in combinations(sorted(words_per_topic), 2):
            #wp_npmi = pmi.npmi(w1, w2)
            wp_npmi = npmi.wd_id_pair_npmi(w1, w2)
            if verbose:
                print("NPMI({}, {}) = {}".format(
                    inference_model.vocab.idx_to_token[w1],
                    inference_model.vocab.idx_to_token[w2],
                    wp_npmi)
                )
            total_topic_npmi += wp_npmi
        total_topic_npmi *= (2 / (N * (N-1)))
        print("Topic {}, NPMI = {}".format(i, total_topic_npmi))
        total_npmi += total_topic_npmi
    print("**** FINAL NPMI = {} *******".format(total_npmi / len(top_k_words_per_topic)))
              


