# coding: utf-8

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock

class BowNTM(HybridBlock):

    def __init__(self, batch_size, vocab_size, l1_dim, n_latent, gen_layers=3, ctx=mx.cpu(), prefix=None, params=None):
        super(BowNTM, self).__init__(prefix=prefix, params=params)
        self.batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx = ctx
        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential()
            self.encoder.add(gluon.nn.Dense(units = l1_dim, activation='tanh'))
            self.encoder.add(gluon.nn.Dense(units = n_latent*2, activation=None))
            self.generator = gluon.nn.HybridSequential()
            with self.generator.name_scope():
                for i in range(gen_layers):
                    self.generator.add(gluon.nn.Dense(units=n_latent, activation='tanh'))
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=vocab_size, activation='tanh')


    def hybrid_forward(self, F, data):
        ## data should have shame N x V - but be SPARSE
        enc_out = self.encoder(data)
        mu_lv = F.split(enc_out, axis=1, num_outputs=2) ## split in half along final dimension
        mu = mu_lv[0]  ## mean
        lv = mu_lv[1]  ## log of the variance
        ## Standard Gaussian VAE - using reparameterization trick
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=self.model_ctx)
        ## z is the (single) sample 
        z = mu + F.exp(0.5*lv) * eps        

        ## generate and decode
        gen_out = self.generator(z)  
        res = gen_out + z
        dec_out = self.decoder(res)
        y = F.log_softmax(dec_out)

        ## L1 regularization penalty on decoder weights
        dec_weights = self.decoder.params.get('weight').data() ## use .var() for a symbol
        l1_weights = F.broadcast_to(F.sum(F.abs(dec_weights)), (self.batch_size,))
        
        ce_loss = -F.sparse.sum(data * y, axis=(-1))
        KL = 0.5 * F.sum(1 + lv - mu*mu - F.exp(lv), axis=1)
        return l1_weights + ce_loss - KL, ce_loss, l1_weights
        
