#coding: utf-8

from mxnet import gluon as nn

__all__ = ['LatentDistribution']

class LatentDistribution(nn.HybridBlock):

    def __init__(self, n_latent, ctx):
        super(LatentDistribution, self).__init__()
        self.n_latent = n_latent
        self.model_ctx = ctx

    ## this is required by most priors
    def _get_gaussian_sample(self, F, mu, lv, batch_size):
        eps = F.random_normal(loc=0, scale=1, shape=(batch_size, self.n_latent), ctx=self.model_ctx)
        return mu + F.exp(0.5*lv) * eps

    ## this is required by most priors
    def _get_unit_var_gaussian_sample(self, F, mu, batch_size):
        eps = F.random_normal(loc=0, scale=1, shape=(batch_size, self.n_latent), ctx=self.model_ctx)
        return mu + eps
    
    
