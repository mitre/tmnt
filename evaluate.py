#!/usr/bin/env python3

import math
import argparse

import mxnet as mx

from tmnt.bow_vae.runtime import BowNTMInference
from tmnt.coherence.pmi import PMI
from tmnt.utils.ngram_helpers import UnigramReader, BigramReader

from itertools import combinations

from pprint import pprint
from pathlib import Path

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Variational AutoEncoder topic model')
    parser.add_argument('--gpu', type=int, help='GPU device ID (-1 default = CPU)', default=-1)
    parser.add_argument('--vec_file', type=str, required=True, help='ile in sparse vector format')
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file associated with sparse vector data')
    parser.add_argument('--model_dir', required=True, type=Path,
                        help='The directory where the params, specs, and vocab should be found.')
    return parser

def get_top_k_words_per_topic(inference, num_topics, k):
    w = inference.model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
    return [[inference.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy())] for t in range(num_topics)]

def get_top_k_word_idx_per_topic(inference, num_topics, k):
    w = inference.model.decoder.collect_params().get('weight').data()
    sorted_ids = w.argsort(axis=0, is_ascend=False)
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

    param_file, specs_file, vocab_file = \
        args.model_dir / "model.params", args.model_dir / "model.specs", args.model_dir / "vocab.json"

    inference_model = BowNTMInference(param_file, specs_file, vocab_file,
                                      ctx=mx.cpu() if args.gpu < 0 else mx.gpu(args.gpu))

    top_k_words_per_topic = get_top_k_word_idx_per_topic(inference_model, 5, 5)

    unigram_reader = UnigramReader(args.vocab_file)
    bigram_reader = BigramReader(args.vec_file)

    pmi = PMI(unigram_reader.unigrams, bigram_reader.bigrams)

    for i, words_per_topic in enumerate(top_k_words_per_topic):
        print("Topic {0}".format(i))
        for (w1, w2) in combinations(words_per_topic, 2):
            print("NPMI({0}, {1}) = {2}".format(
                inference_model.vocab.idx_to_token[w1],
                inference_model.vocab.idx_to_token[w2],
                pmi.npmi(w1, w2))
            )


