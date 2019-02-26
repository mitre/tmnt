# coding: utf-8

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
from tmnt.distributions import LogisticGaussianLatentDistribution, GaussianLatentDistribution, HyperSphericalLatentDistribution
import numpy as np
import math

__all__ = ['BowNTM', 'MetaDataBowNTM']



class BowNTM(HybridBlock):
    """
    Parameters
    ----------
    vocabulary : int size of the vocabulary
    enc_dim : int number of dimension of input encoder (first FC layer)
    n_latent : int number of dimensions of the latent dimension (i.e. number of topics)
    gen_layers : int (default = 3) number of generator layers (after sample); size is the same as n_latent
    batch_size : int (default None) provided only at training time (or when model is Hybridized) - otherwise will be inferred
    ctx : context device (default is mx.cpu())
    """
    def __init__(self, vocabulary, enc_dim, n_latent, embedding_size, fixed_embedding=False, latent_distrib='logistic_gaussian',
                 init_l1=0.0, coherence_reg_penalty=0.0, batch_size=None, wd_freqs=None, ctx=mx.cpu()):
        super(BowNTM, self).__init__()
        self.batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.vocab_size = len(vocabulary)
        self.coherence_reg_penalty = coherence_reg_penalty
        self.embedding_size = embedding_size
        if vocabulary.embedding:
            assert vocabulary.embedding.idx_to_vec[0].size == embedding_size
        with self.name_scope():
            self.l1_pen_const = self.params.get('l1_pen_const',
                                      shape=(1,),
                                      init=mx.init.Constant([init_l1]), 
                                      differentiable=False)
            self.embedding = gluon.nn.Dense(in_units=self.vocab_size, units=self.embedding_size, activation='tanh')
            self.encoder = gluon.nn.Dense(units = enc_dim, activation='softrelu') ## just single FC layer 'encoder'
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=100.0, ctx=self.model_ctx)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx)                
            self.post_sample_dr_o = gluon.nn.Dropout(0.2)
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.vocab_size, activation=None)
            self.coherence_regularization = CoherenceRegularizer(coherence_reg_penalty)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        if vocabulary.embedding:            
            emb = vocabulary.embedding.idx_to_vec.transpose()
            #g_noise = mx.nd.random.normal(loc=0.0, scale=0.1, shape=emb_1.shape)
            #emb = emb_1 + g_noise
            emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=0) + 1e-10
            emb_norm = emb / emb_norm_val
            #print("Emb norm = {} with sum = {}".format(emb_norm, emb_norm.sum()))
            self.embedding.weight.set_data(emb_norm)
            if fixed_embedding:
                self.embedding.collect_params().setattr('grad_req', 'null')
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
        mu = self.latent_dist.mu_encoder(self.encoder(self.embedding(data)))
        norm = mx.nd.norm(mu, axis=1, keepdims=True)
        return mu / norm
    
    def get_l1_penalty_term(self, F, l1_pen_const, batch_size):
        if F is mx.ndarray:
            dec_weights = self.decoder.params.get('weight').data()
        else:
            dec_weights = self.decoder.params.get('weight').var()
        return l1_pen_const * F.sum(F.abs(dec_weights))
        

    def hybrid_forward(self, F, data, l1_pen_const=None):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        emb_out = self.embedding(data)
        enc_out = self.encoder(emb_out)
        z, KL = self.latent_dist(enc_out, batch_size)
        z_do = self.post_sample_dr_o(z)
        
        res = F.softmax(z_do)
        dec_out = self.decoder(res)
        
        y = F.softmax(dec_out, axis=1)

        l1_pen = self.get_l1_penalty_term(F, l1_pen_const, batch_size)
        recon_loss = -F.sparse.sum( data * F.log(y+1e-12))

        i_loss = recon_loss + l1_pen + KL
        
        ## coherence-focused regularization term
        if self.coherence_reg_penalty > 0.0:
            if F is mx.ndarray:
                w = self.decoder.params.get('weight').data()
                emb = self.embedding.params.get('weight').data()
            else:
                w = self.decoder.params.get('weight').var()
                emb = self.embedding.params.get('weight').var()
            c = self.coherence_regularization(w, emb) * self.coherence_reg_penalty
            final_loss = i_loss + c
        else:
            final_loss = i_loss
        return final_loss, recon_loss, l1_pen, y


class MetaDataBowNTM(BowNTM):

    def __init__(self, n_covars, vocabulary, enc_dim, n_latent, embedding_size, latent_distrib='logistic_gaussian',
                 init_l1=0.0, coherence_reg_penalty=0.0, batch_size=None, wd_freqs=None, ctx=mx.cpu()):
        super(MetaDataBowNTM, self).__init__(vocabulary, enc_dim, n_latent, embedding_size, latent_distrib, init_l1, coherence_reg_penalty, batch_size, wd_freqs, ctx)
        self.n_covars = n_covars
        with self.name_scope():
            self.cov_decoder = CovariateModel(self.n_latent, self.n_covars, self.vocab_size, batch_size=self.batch_size, interactions=False)

    def hybrid_forward(self, F, data, labels, l1_pen_const=None):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        emb_out = self.embedding(data)
        enc_out = self.encoder(F.concat(emb_out, labels))

        z, KL = self.latent_dist(enc_out, batch_size)
        z_do = self.post_sample_dr_o(z)
        res = F.softmax(z_do)
        dec_out = self.decoder(res)
        cov_dec_out = self.cov_decoder(res, labels)
        y = F.softmax(dec_out + cov_dec_out, axis=1)

        ###  Lots of cut and pasting ... refactor this!!
        l1_pen = self.get_l1_penalty_term(F, l1_pen_const, batch_size)
        recon_loss = -F.sparse.sum( data * F.log(y+1e-12))
        i_loss = recon_loss + l1_pen + KL
        
        ## coherence-focused regularization term
        if self.coherence_reg_penalty > 0.0:
            if F is mx.ndarray:
                w = self.decoder.params.get('weight').data()
                emb = self.embedding.params.get('weight').data()
            else:
                w = self.decoder.params.get('weight').var()
                emb = self.embedding.params.get('weight').var()
            c = self.coherence_regularization(w, emb) * self.coherence_reg_penalty
            final_loss = i_loss + c
        else:
            final_loss = i_loss
        return final_loss, recon_loss, l1_pen, y
    
        
class CovariateModel(HybridBlock):

    def __init__(self, n_topics, n_covars, vocab_size, batch_size=None, interactions=False, ctx=mx.cpu()):
        self.n_topics = n_topics
        self.n_covars = n_covars
        self.vocab_size = vocab_size
        self.interactions = interactions
        self.batch_size = batch_size
        self.model_ctx = ctx
        super(CovariateModel, self).__init__()
        with self.name_scope():
            self.cov_decoder = gluon.nn.Dense(in_units=n_covars, units=self.vocab_size, activation=None, use_bias=False)
            if self.interactions:
                self.cov_inter_decoder = gluon.nn.Dense(in_units = self.n_covars * self.n_topics, units=self.vocab_size, activation=None, use_bias=False)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
                

    def hybrid_forward(self, F, topic_distrib, covars):
        score_C = self.cov_decoder(covars)
        if self.interactions:
            td_rsh = F.expand_dims(topic_distrib, 2)
            cov_rsh = F.expand_dims(covars, 1)
            cov_interactions = td_rsh * cov_rsh
            batch_size = cov_interactions.shape[0] if F is mx.ndarray else self.batch_size
            cov_interactions_rsh = F.reshape(cov_interactions, (batch_size, self.n_topics * self.n_covars))
            score_CI = self.cov_inter_decoder(cov_interactions_rsh)
            return score_C + score_CI
        else:
            return score_C
            
    

class CoherenceRegularizer(HybridBlock):

    ## Follows paper: http://aclweb.org/anthology/D18-1096

    def __init__(self, regularizer_pen):
        super(CoherenceRegularizer, self).__init__()
        self.regularizer_pen = regularizer_pen
        

    def hybrid_forward(self, F, w, emb):
        ## emb should have shape (V x D)
        ## w should have shape (V x K)
        # w NORM over columns
            
        w_norm_val = F.norm(w, keepdims=True, axis=0)
        #emb_norm_val = F.norm(emb, keepdims=True, axis=1)
        
        w_norm = F.broadcast_div(w, w_norm_val)
        #emb_norm = F.broadcast_div(emb, emb_norm_val)
        emb_norm = emb  ## assume this is normalized up front (esp. if fixed)

        T = F.linalg.gemm2(emb_norm, w_norm)
        
        T_norm_vals = F.norm(T, keepdims=True, axis=0)
        T_norm = F.broadcast_div(T, T_norm_vals)
        S = F.linalg.gemm2(F.transpose(emb_norm), T_norm) # (V x K)
        C = F.sum(S * w_norm)
        return C
        
