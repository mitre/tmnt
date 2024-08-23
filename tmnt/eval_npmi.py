# coding: utf-8
# Copyright (c) 2019 The MITRE Corporation.
"""
Utilities for computing coherence based on Normalized Pointwise Mutual Information (NPMI).
"""

from math import log10
from collections import Counter

import numpy as np
import scipy
import scipy.sparse
from tqdm import tqdm

from tmnt.utils.ngram_helpers import BigramReader
from itertools import combinations
from gensim.models.coherencemodel import CoherenceModel
from tmnt.preprocess.vectorizer import TMNTVectorizer
from gensim.corpora.dictionary import Dictionary

__all__ = ['NPMI', 'EvaluateNPMI']

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


class EvaluateNPMI(object):

    def __init__(self, top_k_words_per_topic):
        self.top_k_words_per_topic = top_k_words_per_topic

    def evaluate_sp_vec(self, test_sparse_vec):
        reader = BigramReader(test_sparse_vec)
        npmi = NPMI(reader.unigrams, reader.bigrams, reader.n_docs)
        total_npmi = 0
        for i, words_per_topic in enumerate(self.top_k_words_per_topic):
            total_topic_npmi = 0
            N = len(words_per_topic)
            for (w1, w2) in combinations(sorted(words_per_topic), 2):
                wp_npmi = npmi.wd_id_pair_npmi(w1, w2)
                total_topic_npmi += wp_npmi
            total_topic_npmi *= (2 / (N * (N-1)))
            total_npmi += total_topic_npmi
        return total_npmi / len(self.top_k_words_per_topic)

    def evaluate_csr_mat(self, csr_mat):
        if isinstance(csr_mat, scipy.sparse.csr.csr_matrix):
            is_sparse = True
            mat = csr_mat
        else:
            #is_sparse = isinstance(csr_mat, mx.nd.sparse.CSRNDArray)
            is_sparse = False
            if is_sparse:
                mat = csr_mat.asscipy()
            else:
                mat = csr_mat.to_dense().cpu().numpy()
        n_docs = mat.shape[0]
        total_npmi = 0
        for i, words_per_topic in enumerate(self.top_k_words_per_topic):
            total_topic_npmi = 0
            n_topics = len(words_per_topic)
            for (w1, w2) in combinations(sorted(words_per_topic), 2):
                o_1 = mat[:, w1] > 0
                o_2 = mat[:, w2] > 0
                if is_sparse:
                    o_1 = o_1.toarray().squeeze()
                    o_2 = o_2.toarray().squeeze()
                occur_1 = np.array(o_1, dtype='int')
                occur_2 = np.array(o_2, dtype='int')
                unigram_1 = occur_1.sum()
                unigram_2 = occur_2.sum()
                bigram_cnt = np.sum(occur_1 * occur_2)
                if bigram_cnt < 1:
                    npmi = 0.0
                else:
                    npmi = (log10(n_docs) + log10(bigram_cnt) - log10(unigram_1) - log10(unigram_2)) / (log10(n_docs) - log10(bigram_cnt) + 1e-4)
                total_topic_npmi += npmi
            total_topic_npmi *= (2 / (n_topics * (n_topics-1)))
            total_npmi += total_topic_npmi
        return total_npmi / len(self.top_k_words_per_topic)

    def get_full_vocab_npmi_matrix(self, mat):
        vocab_size = mat.shape[1]
        npmi_matrix = np.zeros((vocab_size, vocab_size))
        n_docs = mat.shape[0]
        if isinstance(mat, scipy.sparse.csr.csr_matrix):
            is_sparse = True
        for (w1, w2) in tqdm(combinations(np.arange(vocab_size), 2)):
            o_1 = mat[:, w1] > 0
            o_2 = mat[:, w2] > 0
            if is_sparse:
                o_1 = o_1.toarray().squeeze()
                o_2 = o_2.toarray().squeeze()
            occur_1 = np.array(o_1, dtype='int')
            occur_2 = np.array(o_2, dtype='int')
            unigram_1 = occur_1.sum()
            unigram_2 = occur_2.sum()
            bigram_cnt = np.sum(occur_1 * occur_2)
            if bigram_cnt < 1:
                npmi = 0.0
            else:
                npmi = (log10(n_docs) + log10(bigram_cnt) - log10(unigram_1) - log10(unigram_2)) / (log10(n_docs) - log10(bigram_cnt) + 1e-4)
            npmi_matrix[w1, w2] = npmi
        return npmi_matrix
    
class EvaluateNPMIUmass(object):

    def __init__(self, npmi_matrix: np.array, vectorizer: TMNTVectorizer):
        self.vectorizer = vectorizer
        self.npmi_matrix = npmi_matrix # by convention this will be lower-triangular
        dim = npmi_matrix.shape[0]
        for mc in range(self.npmi_matrix.shape[0]):
            for i in range(mc+1,dim):
                self.npmi_matrix[mc,i] = self.npmi_matrix[i,mc]
    
    def evaluate_topics(self, topic_ids):
        npmi_score = 0.0
        total_size = len(topic_ids) * len(topic_ids[0])
        for topic in topic_ids:
            for (w1, w2) in combinations(topic):
                npmi_score += self.npmi_matrix[w1, w2]
        return npmi_score / total_size



class FullNPMI(object):

    def get_full_vocab_npmi_matrix(self, mat: scipy.sparse.csr_matrix, tf: TMNTVectorizer):
        corpus = []
        npmi_matrix = np.zeros((tf.vocab_size, tf.vocab_size))
        for ri in range(mat.shape[0]):
            row = mat.getrow(ri)
            corpus.append(list(zip(row.indices, row.data)))
        topics = [ list(range(mat.shape[1])) ]
        dictionary = Dictionary()
        dictionary.id2token = tf.get_vocab().get_itos()
        dictionary.token2id = tf.get_vocab().get_stoi()
        cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass', topn=len(topics[0])) 
        segmented_topics = cm.measure.seg(cm.topics)
        accumulator = cm.estimate_probabilities(segmented_topics)
        num_docs = accumulator.num_docs
        eps = 1e-12
        for w1, w2 in tqdm(segmented_topics[0]):
            w1_count = accumulator[w1]
            w2_count = accumulator[w2]
            co_occur_count = accumulator[w1, w2]
            p_w1_w2 = co_occur_count / num_docs
            p_w1 = w1_count / num_docs
            p_w2 = w2_count / num_docs
            npmi_matrix[w1, w2] = np.log((p_w1_w2 + eps) / (p_w1 * p_w2)) / -np.log(p_w1_w2  + eps)
        return npmi_matrix



