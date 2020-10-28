# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""


__all__ = ['BaseInferencer']

class BaseInferencer(object):
    """Base inference object for text encoding with a trained topic model.

    """
    def __init__(self, ctx):
        self.ctx = ctx

    def encode_texts(self, intexts):
        raise NotImplementedError

    def get_top_k_words_per_topic(self, k):
        raise NotImplementedError

    def get_top_k_words_per_topic_per_covariate(self, k):
        raise NotImplementedError


class BaseTextEncoder(object):
    """Base text encoder for various topic models

    Args:
        inferencer (`tmnt.models.base.base_inference.BaseInferencer`): Inferencer object that runs the encoder portion of a VAE/VED model.
        use_probs (bool): Map topic vector encodings to the simplex (sum to 1).
        temperature (float): Temperature to sharpen/flatten encoding distribution.
    """
    def __init__(self, inferencer, use_probs=True, temperature=0.5):
        self.temp      = temperature
        self.inference = inference
        self.use_probs = use_probs

    def encode_single_string(self, txt_string):
        raise NotImplementedError

    def encode_batch(self, txtx, covars=None, pool_size=4):
        raise NotImplementedError

    
