#coding: utf-8

import math
import mxnet as mx
from scipy import special as sp
from mxnet import gluon
from tmnt.distributions import LatentDistribution

__all__ = ['HyperSphericalUniform']

class HyperSphericalUniform(LatentDistribution):

    def __init__(self, dim, ctx):
        super(HyperSphericalUniform, self).__init__(n_latent, ctx)
        self.kappa = 1.0
        
        with self.name_scope():
            self.mu_encoder = gluon.nn.Dense(units = n_latent, activation=None)


    def _log_surface_area(self):
        return math.log(2) + ((self.n_latent + 1) / 2) * math.log(math.pi) - \
            mx.nd.gammaln(mx.nd.array([(self.n_latent + 1) / 2], ctx=self.ctx))


    def hybrid_forward(self, F, data):
        mu = self.mu_encoder(data)
        norm = F.norm(mu, axis=1)
        mu = mu / norm
        return mu

    @staticmethod
    def _vmf_kld(k, d):
        tmp = (k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) \
               + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k)) \
               - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2).real
        if tmp != tmp:
            exit()
        return np.array([tmp])

