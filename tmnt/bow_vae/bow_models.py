# coding: utf-8

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock

__all__ = ['BowNTM']



class BowNTM(HybridBlock):

    def __init__(self, batch_size, vocab_size, l1_dim, n_latent, gen_layers=3, ctx=mx.cpu()):
        super(BowNTM, self).__init__()
        self.batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx = ctx

        with self.name_scope():
            self.l1_pen_const = self.params.get('l1_pen_const',
                                      shape=(1,),
                                      init=mx.init.Constant([0.001]), 
                                      differentiable=False)
            self.encoder = gluon.nn.Dense(units = l1_dim, activation='tanh') ## just single FC layer
            self.mu_encoder = gluon.nn.Dense(units = n_latent, activation=None)
            self.lv_encoder = gluon.nn.Dense(units = n_latent, activation='softrelu')
            self.generator = gluon.nn.HybridSequential()
            with self.generator.name_scope():
                for i in range(gen_layers):
                    self.generator.add(gluon.nn.Dense(units=n_latent, activation='tanh'))
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=vocab_size, activation=None)

    def get_gaussian_sample(mu, lv):
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=self.model_ctx)
        z = mu + F.exp(0.5*lv) * eps
        return z

    def get_gaussian_kl(mu, lv):
        return 0.5 * F.sum(1 + lv - mu*mu - F.exp(lv), axis=1)

    def get_vmf_sample(mu, lv):
        
        

    def hybrid_forward(self, F, data, l1_pen_const):
        enc_out = self.encoder(data)
        mu = self.mu_encoder(enc_out)
        lv = self.lv_encoder(enc_out)
        ## Standard Gaussian VAE - using reparameterization trick
        z = get_gaussian_sample(mu, lv)

        ## generate and decode
        gen_out = self.generator(z)  
        res = gen_out + z
        dec_out = self.decoder(res)
        y = F.log_softmax(dec_out)

        ## L1 regularization penalty on decoder weights
        if F is mx.ndarray:
            dec_weights = self.decoder.params.get('weight').data()
        else:
            dec_weights = self.decoder.params.get('weight').var()
        l1_weights_1 = l1_pen_const * F.sum(F.abs(dec_weights))
        l1_weights = F.broadcast_to(l1_weights_1, (self.batch_size,))
        
        ce_loss = -F.sparse.sum(data * y, axis=(-1))
        KL = get_gaussian_kl(mu, lv)

        return l1_weights + ce_loss - KL, ce_loss, l1_weights
        
