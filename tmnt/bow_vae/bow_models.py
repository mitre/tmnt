# coding: utf-8

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
import numpy as np
import math

__all__ = ['BowNTM', 'RichGeneratorBowNTM']



class BowNTM(HybridBlock):
    """
    Parameters
    ----------
    vocab_size : int size of the vocabulary
    enc_dim : int number of dimension of input encoder (first FC layer)
    n_latent : int number of dimensions of the latent dimension (i.e. number of topics)
    gen_layers : int (default = 3) number of generator layers (after sample); size is the same as n_latent
    batch_size : int (default None) provided only at training time (or when model is Hybridized) - otherwise will be inferred
    ctx : context device (default is mx.cpu())
    """
    def __init__(self, vocab_size, enc_dim, n_latent, batch_size=None, wd_freqs=None, ctx=mx.cpu()):
        super(BowNTM, self).__init__()
        self.batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.alpha = 1.0
        self.vocab_size = vocab_size
        
        prior_var = 1 / self.alpha - (2.0 / n_latent) + 1 / (self.n_latent * self.n_latent)
        self.prior_var = prior_var
        self.prior_logvar = math.log(prior_var)

        with self.name_scope():
            self.l1_pen_const = self.params.get('l1_pen_const',
                                      shape=(1,),
                                      init=mx.init.Constant([0.001]), 
                                      differentiable=False)
            self.encoder = gluon.nn.Dense(units = enc_dim, activation='softrelu') ## just single FC layer 'encoder'
            self.mu_encoder = gluon.nn.Dense(units = n_latent, activation=None)
            self.lv_encoder = gluon.nn.Dense(units = n_latent, activation=None)
            self.post_sample_dr_o = gluon.nn.Dropout(0.2)
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.vocab_size, activation=None)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        ## Initialize decoder bias terms to corpus frequencies
        if wd_freqs:
            freq_nd = mx.nd.array(wd_freqs, ctx=ctx) + 1
            total = freq_nd.sum()
            log_freq = freq_nd.log() - freq_nd.sum().log()
            bias_param = self.decoder.collect_params().get('bias')
            bias_param.set_data(log_freq)
            bias_param.grad_req = 'null'
            self.out_bias = bias_param.data()

            
    def encode_data(self, data):
        return self.mu_encoder(self.encoder(data))
    
            
    def get_gaussian_sample(self, F, mu, lv, batch_size):
        eps = F.random_normal(loc=0, scale=1, shape=(batch_size, self.n_latent), ctx=self.model_ctx)
        z = mu + F.exp(0.5*lv) * eps
        return self.post_sample_dr_o(z) ## DROPOUT to further reduce co-adaptation in decoder weights

    def get_gaussian_kl(self, F, mu, lv):
        return -0.5 * F.sum(1 + lv - mu*mu - F.exp(lv), axis=1)


    def get_logistic_normal_kl(self, F, mu, lv):
        posterior_var = F.exp(lv)
        delta = mu
        dt = delta * delta / self.prior_var
        v_div = posterior_var / self.prior_var
        lv_div = self.prior_logvar - lv
        return F.sum(0.5 * (F.sum((v_div + dt + lv_div), axis=1) - self.n_latent))


    def get_mean_and_logvar(self, F, data):        
        enc_out = self.encoder(data)
        mu = self.mu_encoder(enc_out)
        lv = self.lv_encoder(enc_out)
        return mu, lv
    

    def get_l1_penalty_term(self, F, l1_pen_const, batch_size):
        if F is mx.ndarray:
            dec_weights = self.decoder.params.get('weight').data()
        else:
            dec_weights = self.decoder.params.get('weight').var()
        return l1_pen_const * F.sum(F.abs(dec_weights))
        

    def hybrid_forward(self, F, data, l1_pen_const):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        mu, lv = self.get_mean_and_logvar(F, data)
        z = self.get_gaussian_sample(F, mu, lv, batch_size)

        res = F.softmax(z)
        dec_out = self.decoder(res)
        y = F.softmax(dec_out, axis=1)

        l1_pen = self.get_l1_penalty_term(F, l1_pen_const, batch_size)
        recon_loss = -F.sparse.sum( data * F.log(y+1e-12))
        KL = self.get_logistic_normal_kl(F, mu, lv)

        return recon_loss+l1_pen+KL, recon_loss, l1_pen, y
        

class RichGeneratorBowNTM(BowNTM):
    """
    Adds a series of generator FC layers between latent code and decoder/output layer.

    Parameters
    ----------
    vocab_size : int size of the vocabulary
    enc_dim : int number of dimension of input encoder (first FC layer)
    n_latent : int number of dimensions of the latent dimension (i.e. number of topics)
    gen_layers : int (default = 3) number of generator layers (after sample); size is the same as n_latent
    batch_size : int (default None) provided only at training time (or when model is Hybridized) - otherwise will be inferred
    ctx : context device (default is mx.cpu())
    """
    def __init__(self, vocab_size, enc_dim, n_latent, gen_layers=3, batch_size=None, wd_freqs=None, ctx=mx.cpu()):
        super(RichGeneratorBowNTM, self).__init__(vocab_size, enc_dim, n_latent, batch_size=batch_size, wd_freqs=wd_freqs, ctx=ctx)

        self.gen_layers = gen_layers
        with self.name_scope():
            self.generator = gluon.nn.HybridSequential()
            with self.generator.name_scope():
                for i in range(gen_layers):
                    self.generator.add(gluon.nn.Dense(units=n_latent, activation='relu'))


    def hybrid_forward(self, F, data, l1_pen_const):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        mu, lv = self.get_mean_and_logvar(F, data)
        z = self.get_gaussian_sample(F, mu, lv, batch_size)

        gen_out = self.generator(z)  
        res_1 = gen_out + z if self.gen_layers > 1 else gen_out
        res = F.softmax(res_1)
        dec_out = self.decoder(res)
        y = F.softmax(dec_out, axis=1)
        
        # loss terms
        l1_pen = self.get_l1_penalty_term(F, l1_pen_const, batch_size)
        recon_loss = -F.sparse.sum( data * F.log(y+1e-12))
        KL = self.get_logistic_normal_kl(F, mu, lv)

        return recon_loss + KL + l1_pen, recon_loss, l1_pen, y

