#coding: utf-8

import math
import mxnet as mx
from mxnet import gluon
from tmnt.distributions import LatentDistribution

__all__ = ['GaussianLatentDistribution']

class GaussianLatentDistribution(LatentDistribution):

    def __init__(self, n_latent, ctx):
        super(GaussianLatentDistribution, self).__init__(n_latent, ctx)
        with self.name_scope():
            self.mu_encoder = gluon.nn.Dense(units = n_latent, activation=None)
            self.lv_encoder = gluon.nn.Dense(units = n_latent, activation=None)

    def _get_kl_term(self, F, mu, lv):
        return -0.5 * F.sum(1 + lv - mu*mu - F.exp(lv), axis=1)

    def hybrid_forward(self, F, data, batch_size):
        mu = self.mu_encoder(data)
        lv = self.lv_encoder(data)
        z = self._get_gaussian_sample(F, mu, lv, batch_size)
        KL = self._get_kl_term(F, mu, lv)
        return z, KL

