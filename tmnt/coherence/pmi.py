from math import log2, log10
from collections import Counter

import numpy as np

class PMI(object):
    def __init__(self, unigram_freq: Counter, bigram_freq: Counter):
        self.unigram_freq = unigram_freq
        self.bigram_freq = bigram_freq

        self.num_unigrams = np.float32(sum(unigram_freq.values()))
        self.num_bigrams = np.float32(sum(bigram_freq.values()))

    def pmi(self, w1: str, w2: str):
        """
        PMI(w_i, w_j) = log_2(p(w_i, w_j) / (p(w_i) * p(w_j)))
        """
        pw1 = self.unigram_freq[w1] / self.num_unigrams
        pw2 = self.unigram_freq[w2] / self.num_unigrams
        pw1_w2 = self.bigram_freq[(w1, w2)] / self.num_bigrams
        return log2(pw1_w2 / pw1 / pw2)

    def npmi(self, w1: str, w2: str):
        """
        NPMI(w_i, w_j) = PMI(w_i, w_j) / -log_10(p(w_i, w_j))
        """
        pmi = self.pmi(w1, w2)
        pw1_w2 = self.bigram_freq[(w1, w2)] / self.num_bigrams
        return pmi / -log10(pw1_w2)
