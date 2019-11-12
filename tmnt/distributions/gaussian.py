#coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""


from mxnet import gluon
from tmnt.distributions import LatentDistribution

__all__ = ['GaussianLatentDistribution']

class GaussianLatentDistribution(LatentDistribution):

    def __init__(self, n_latent, ctx, dr=0.2):
        super(GaussianLatentDistribution, self).__init__(n_latent, ctx)
        with self.name_scope():
            self.mu_encoder = gluon.nn.Dense(units = n_latent)
            self.lv_encoder = gluon.nn.Dense(units = n_latent)
            self.mu_bn = gluon.nn.BatchNorm(momentum = 0.001, epsilon=0.001)
            self.lv_bn = gluon.nn.BatchNorm(momentum = 0.001, epsilon=0.001)
            self.post_sample_dr_o = gluon.nn.Dropout(dr)
        self.mu_bn.collect_params().setattr('grad_req', 'null')
        self.lv_bn.collect_params().setattr('grad_req', 'null')        

    def _get_kl_term(self, F, mu, lv):
        return -0.5 * F.sum(1 + lv - mu*mu - F.exp(lv), axis=1)

    def hybrid_forward(self, F, data, batch_size):
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)
        lv = self.lv_encoder(data)
        lv_bn = self.lv_bn(lv)
        z = self._get_gaussian_sample(F, mu_bn, lv_bn, batch_size)
        KL = self._get_kl_term(F, mu_bn, lv_bn)
        z = self.post_sample_dr_o(z)
        #z = F.relu(z)
        return z, KL

