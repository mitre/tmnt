#coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""


import math
import mxnet as mx
from mxnet import gluon
from tmnt.distributions.latent_distrib import LatentDistribution

__all__ = ['LogisticGaussianLatentDistribution']

class LogisticGaussianLatentDistribution(LatentDistribution):

    def __init__(self, n_latent, ctx, dr=0.2, alpha=1.0):
        super(LogisticGaussianLatentDistribution, self).__init__(n_latent, ctx)
        self.alpha = alpha

        prior_var = 1 / self.alpha - (2.0 / n_latent) + 1 / (self.n_latent * self.n_latent)
        self.prior_var = mx.nd.array([prior_var], ctx=ctx)
        self.prior_logvar = mx.nd.array([math.log(prior_var)], ctx=ctx)

        with self.name_scope():
            self.mu_encoder = gluon.nn.Dense(units = n_latent)
            self.lv_encoder = gluon.nn.Dense(units = n_latent)
            self.mu_bn = gluon.nn.BatchNorm(momentum = 0.001, epsilon=0.001)
            self.lv_bn = gluon.nn.BatchNorm(momentum = 0.001, epsilon=0.001)
            self.post_sample_dr_o = gluon.nn.Dropout(dr)
        self.mu_bn.collect_params().setattr('grad_req', 'null')
        self.lv_bn.collect_params().setattr('grad_req', 'null')        
            

    def _get_kl_term(self, F, mu, lv):
        posterior_var = F.exp(lv)
        delta = mu
        dt = F.broadcast_div(delta * delta, self.prior_var)
        v_div = F.broadcast_div(posterior_var, self.prior_var)
        lv_div = self.prior_logvar - lv
        return 0.5 * (F.sum((v_div + dt + lv_div), axis=1) - self.n_latent)

    def hybrid_forward(self, F, data, batch_size):
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)        
        lv = self.lv_encoder(data)
        lv_bn = self.lv_bn(lv)
        z_p = self._get_gaussian_sample(F, mu_bn, lv_bn, batch_size)
        KL = self._get_kl_term(F, mu, lv)
        z = self.post_sample_dr_o(z_p)
        return F.softmax(z), KL
