#coding: utf-8

import math
import mxnet as mx

class HyperSphericalUniform(object):

    def __init__(self, dim, ctx):
        self.dim = dim
        self.ctx = ctx

    def get_entropy(self):
        return _log_surface_area(self)

    def _log_surface_area(self):
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - \
            mx.nd.gammaln(mx.nd.array([(self._dim + 1) / 2], ctx=self.ctx))

    def log_prob(self, x):
        return mx.nd.ones(x.shape[:-1], ctx = self.ctx) * self._log_surface_area

    def sample(self, in_shape):
        output = mx.ndarray.random.normal(0, 1, shape=in_shape + (dim+1,))
        return output / output.norm(axis=-1, keepdims=True)
