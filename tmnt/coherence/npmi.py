from math import log2, log10
from collections import Counter

import numpy as np

class NPMI(object):

    def __init__(self, unigram_cnts: Counter, bigram_cnts: Counter, n_docs: int):
        self.unigram_cnts = unigram_cnts
        self.bigram_cnts = bigram_cnts
        self.n_docs = n_docs


    def wd_id_pair_npmi(self, w1: int, w2: int):
        cw1 = self.unigram_cnts.get(w1, 0.0)
        cw2 = self.unigram_cnts.get(w2, 0.0)
        c12 = self.bigram_cnts.get((w1, w2), 0.0)
        if cw1 == 0.0 or cw2 == 0.0 or c12 == 0.0:
            return 0.0
        else:
            return (log10(self.n_docs) + log10(c12) - log10(cw1) - log10(cw2)) / (log10(self.n_docs) - log10(c12))
