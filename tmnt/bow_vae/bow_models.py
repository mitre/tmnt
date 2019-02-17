# coding: utf-8

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
from tmnt.distributions import LogisticGaussianLatentDistribution, GaussianLatentDistribution, HyperSphericalLatentDistribution
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
    def __init__(self, vocab_size, enc_dim, n_latent, latent_distrib='logistic_gaussian',
                 init_l1=0.0, batch_size=None, wd_freqs=None, ctx=mx.cpu()):
        super(BowNTM, self).__init__()
        self.batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.vocab_size = vocab_size
        with self.name_scope():
            self.l1_pen_const = self.params.get('l1_pen_const',
                                      shape=(1,),
                                      init=mx.init.Constant([init_l1]), 
                                      differentiable=False)
            self.encoder = gluon.nn.Dense(units = enc_dim, activation='softrelu') ## just single FC layer 'encoder'
            ## Consider a second encoder layer so the first could be initialized with e.g. word embeddings
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(self.batch_size, n_latent, kappa=100.0, ctx=self.model_ctx)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx)                
            self.post_sample_dr_o = gluon.nn.Dropout(0.2)
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.vocab_size, activation=None)
            self.coherence_regularization = CoherenceRegularizer()
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
        """
        Encode data to the mean of the latent distribution defined by the input `data`
        """
        return self.latent_dist.mu_encoder(self.encoder(data))
    
    def get_l1_penalty_term(self, F, l1_pen_const, batch_size):
        if F is mx.ndarray:
            dec_weights = self.decoder.params.get('weight').data()
        else:
            dec_weights = self.decoder.params.get('weight').var()
        return l1_pen_const * F.sum(F.abs(dec_weights))
        

    def hybrid_forward(self, F, data, l1_pen_const):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        enc_out = self.encoder(data)
        z, KL = self.latent_dist(enc_out, batch_size)
        z_do = self.post_sample_dr_o(z)
        
        res = F.softmax(z)
        dec_out = self.decoder(res)
        y = F.softmax(dec_out, axis=1)

        l1_pen = self.get_l1_penalty_term(F, l1_pen_const, batch_size)
        recon_loss = -F.sparse.sum( data * F.log(y+1e-12))

        ## coherence regularization
        if False:
            if F is mx.ndarray:
                w = self.decoder.params.get('weight').data()
                emb = self.encoder.params.get('weight').data()
            else:
                w = self.decoder.params.get('weight').var()
                emb = self.encoder.params.get('weight').var()
            c = self.coherence_regularization(w, emb) * 10000

        return recon_loss + l1_pen + KL, recon_loss, l1_pen, y
        

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
    def __init__(self, vocab_size, enc_dim, n_latent, latent_distrib='logistic_gaussian', init_l1=0.0,
                 gen_layers=3, batch_size=None, wd_freqs=None, ctx=mx.cpu()):
        super(RichGeneratorBowNTM, self).__init__(vocab_size, enc_dim, n_latent, latent_distrib=latent_distrib, init_l1=init_l1,
                                                  batch_size=batch_size, wd_freqs=wd_freqs, ctx=ctx)
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


class CoherenceRegularizer(HybridBlock):

    ## Follows paper: http://aclweb.org/anthology/D18-1096

    def __init__(self):
        super(CoherenceRegularizer, self).__init__()

    def hybrid_forward(self, F, w, emb):
        ## emb should have shape (V x D)
        ## w should have shape (V x K)
        # w NORM over columns
            
        w_norm_val = F.norm(w, keepdims=True, axis=0)
        emb_norm_val = F.norm(emb, keepdims=True, axis=1)
        
        w_norm = w / w_norm_val
        emb_norm = emb / emb_norm_val

        #print("Shape w_norm = {}, emb_norm = {}".format(w_norm.shape, emb_norm.shape))
        
        T = F.linalg.gemm2(emb_norm, w_norm)
        
        T_norm_vals = F.norm(T, keepdims=True, axis=0)
        T_norm = T / T_norm_vals
        ## T should have shape (D x K)
        #print("Shape T_norm = {}, emb_norm = {}".format(T_norm.shape, emb_norm.shape))
        S = F.linalg.gemm2(F.transpose(emb_norm), T_norm) # (V x K)
        C_mat = S * w_norm
        #C_vec = F.sum(C, axis=0) # some over vocab - i.e. rows
        C = F.sum(C_mat) ## coherence-based regularization
        #print("C = {} val = {}".format(C.shape, C.asscalar()))
        return C
        
