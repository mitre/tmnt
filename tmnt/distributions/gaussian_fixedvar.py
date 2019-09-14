#coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""


from mxnet import gluon
import numpy as np
import mxnet as mx
from tmnt.distributions import LatentDistribution

__all__ = ['GaussianUnitVarLatentDistribution']

class GaussianUnitVarLatentDistribution(LatentDistribution):

    def __init__(self, n_latent, ctx, dr=0.2, var=1.0):
        super(GaussianUnitVarLatentDistribution, self).__init__(n_latent, ctx)
        self.variance = mx.nd.array([var], ctx=ctx)
        self.log_variance = mx.nd.log(self.variance)
        with self.name_scope():
            self.mu_encoder = gluon.nn.Dense(units = n_latent)
            self.mu_bn = gluon.nn.BatchNorm(momentum = 0.001, epsilon=0.001)
            self.post_sample_dr_o = gluon.nn.Dropout(dr)
        self.mu_bn.collect_params().setattr('grad_req', 'null')

    def _get_kl_term(self, F, mu):
        return -0.5 * F.sum(1.0 + self.log_variance - mu*mu - self.variance, axis=1)

    def hybrid_forward(self, F, data, batch_size):
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)
        z = self._get_gaussian_sample(F, mu_bn, self.log_variance, batch_size)
        KL = self._get_kl_term(F, mu_bn)
        return self.post_sample_dr_o(z), KL
