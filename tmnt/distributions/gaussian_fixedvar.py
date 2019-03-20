#coding: utf-8

import math
import mxnet as mx
from mxnet import gluon
from tmnt.distributions import LatentDistribution

__all__ = ['GaussianUnitVarLatentDistribution']

class GaussianUnitVarLatentDistribution(LatentDistribution):

    def __init__(self, n_latent, ctx, dr=0.2):
        super(GaussianUnitVarLatentDistribution, self).__init__(n_latent, ctx)
        with self.name_scope():
            self.mu_encoder = gluon.nn.Dense(units = n_latent, activation=None)
            self.mu_bn = gluon.nn.BatchNorm(momentum = 0.001, epsilon=0.001)
            self.post_sample_dr_o = gluon.nn.Dropout(dr)
        self.mu_bn.collect_params().setattr('grad_req', 'null')

    def _get_kl_term(self, F, mu):
        return -0.5 * F.sum(-mu*mu, axis=1)

    def hybrid_forward(self, F, data, batch_size):
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)
        z = self._get_unit_var_gaussian_sample(F, mu_bn, batch_size)
        KL = self._get_kl_term(F, mu_bn)
        return self.post_sample_dr_o(z), KL
