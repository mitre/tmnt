# coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
from tmnt.distributions import LogisticGaussianLatentDistribution
from tmnt.distributions import GaussianLatentDistribution
from tmnt.distributions import HyperSphericalLatentDistribution
from tmnt.distributions import GaussianUnitVarLatentDistribution
import logging

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
                 init_l1=0.0, coherence_reg_penalty=0.0, kappa=100.0, alpha=1.0, target_sparsity = 0.0, batch_size=None,
                 n_encoding_layers = 3, enc_dr=0.1,
                 wd_freqs=None, seed_mat=None, n_covars=0, ctx=mx.cpu()):
        super(BowNTM, self).__init__()
        self.batch_size = batch_size
        self._orig_batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.vocab_size = len(vocabulary)
        self.coherence_reg_penalty = coherence_reg_penalty
        self.embedding_size = embedding_size
        self.target_sparsity = target_sparsity
        self.vocabulary = vocabulary
        if vocabulary.embedding:
            assert vocabulary.embedding.idx_to_vec[0].size == embedding_size
        self.encoding_dims = [self.embedding_size + n_covars] + [enc_dim for _ in range(n_encoding_layers)]
        with self.name_scope():
            self.l1_pen_const = self.params.get('l1_pen_const',
                                      shape=(1,),
                                      init=mx.init.Constant([init_l1]), 
                                      differentiable=False)
            ## Add in topic seed constraints
            self.seed_matrix = seed_mat
            ## should be tanh here to avoid losing embedding information
            self.embedding = gluon.nn.Dense(in_units=self.vocab_size, units=self.embedding_size, activation='tanh')
            self.encoder = self._get_encoder(self.encoding_dims, dr=enc_dr)
            #self.encoder = gluon.nn.Dense(in_units=(self.embedding_size + n_covars),
            #                              units = enc_dim, activation='softrelu') ## just single FC layer 'encoder'
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx, alpha=alpha)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=kappa, ctx=self.model_ctx)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarLatentDistribution(n_latent, ctx)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.vocab_size, activation=None)
            self.coherence_regularization = CoherenceRegularizer(coherence_reg_penalty)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        if vocabulary.embedding:            
            emb = vocabulary.embedding.idx_to_vec.transpose()
            emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=0) + 1e-10
            emb_norm = emb / emb_norm_val
            self.embedding.weight.set_data(emb_norm)
            if fixed_embedding:
                self.embedding.collect_params().setattr('grad_req', 'null')
        ## Initialize and FIX decoder bias terms to corpus frequencies
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1
            total = freq_nd.sum()
            log_freq = freq_nd.log() - freq_nd.sum().log()
            bias_param = self.decoder.collect_params().get('bias')
            bias_param.set_data(log_freq)
            bias_param.grad_req = 'null'
            self.out_bias = bias_param.data()


    def _get_encoder(self, dims, dr=0.1):
        encoder = gluon.nn.HybridSequential()
        for i in range(len(dims)-1):
            encoder.add(gluon.nn.Dense(in_units=dims[i], units=dims[i+1], activation='softrelu'))
            if dr > 0.0:
                encoder.add(gluon.nn.Dropout(dr))
        return encoder

    def get_top_k_terms(self, k):
        """
        Returns the top K terms for each topic based on sensitivity analysis. Terms whose 
        probability increases the most for a unit increase in a given topic score/probability
        are those most associated with the topic.
        """
        z = mx.nd.ones(shape=(1, self.n_latent), ctx=self.model_ctx)
        jacobian = mx.nd.zeros(shape=(self.vocab_size, self.n_latent), ctx=self.model_ctx)
        z.attach_grad()        
        for i in range(self.vocab_size):
            with mx.autograd.record():
                y = self.decoder(z)
                yi = y[0][i]
            yi.backward()
            jacobian[i] = z.grad
        sorted_j = jacobian.argsort(axis=0, is_ascend=False)
        return sorted_j

    def encode_data(self, data):
        """
        Encode data to the mean of the latent distribution defined by the input `data`
        """
        return self.latent_dist.mu_encoder(self.encoder(self.embedding(data)))
    
    def get_l1_penalty_term(self, F, l1_pen_const, batch_size):
        if F is mx.ndarray:
            dec_weights = self.decoder.params.get('weight').data()
        else:
            dec_weights = self.decoder.params.get('weight').var()
        return l1_pen_const * F.sum(F.abs(dec_weights))

    def add_coherence_reg_penalty(self, F, cur_loss):
        if self.coherence_reg_penalty > 0.0:
            if F is mx.ndarray:
                w = self.decoder.params.get('weight').data()
                emb = self.embedding.params.get('weight').data()
            else:
                w = self.decoder.params.get('weight').var()
                emb = self.embedding.params.get('weight').var()
            c = (self.coherence_regularization(w, emb) * self.coherence_reg_penalty)
            return (cur_loss + c), c
        else:
            #return (cur_loss, None)
            return (cur_loss, F.zeros_like(cur_loss))

    def add_seed_constraint_loss(self, F, cur_loss):
        # G - number of seeded topics
        # S - number of seeds per topic
        # K - number of topics
        if self.seed_matrix is not None:
            if F is mx.ndarray:
                w = self.decoder.params.get('weight').data()
            else:
                w = self.decoder.params.get('weight').var()
            ts = F.take(w, self.seed_matrix)   ## should have shape (G, S, K)
            ts_sums = F.sum(ts, axis=1) # now (G, K)
            ts_probs = F.softmax(ts_sums, axis=1)
            entropies = -F.sum(ts_probs * F.log(ts_probs)) ## want to minimize the entropy here
            ## Ensure seed terms have higher weights
            seed_means = F.mean(ts, axis=1)  # (G,K)
            total_means = F.mean(w, axis=0)  # (K,)
            pref_loss = F.relu(total_means - seed_means) # penalty if mean weight for topic is greater than seed means
            # minimize weighted entropy over the seed means
            seed_pr = F.softmax(seed_means)
            per_topic_entropy = -F.sum(seed_pr * F.log(seed_pr), axis=0)
            seed_means_pr = F.sum(seed_pr, axis=0)
            per_topic_entropy = F.sum(seed_means_pr * per_topic_entropy)
            entropies = F.add(entropies, F.sum(pref_loss))
            entropies = F.add(entropies, per_topic_entropy)
            return (F.broadcast_add(cur_loss, entropies), entropies)
        else:
            return (cur_loss, F.zeros_like(cur_loss))
            #return (cur_loss, None)

    def general_entropy_min_loss(self, F, cur_loss):
        if F is mx.ndarray:
            w = self.decoder.params.get('weight').data()
        else:
            w = self.decoder.params.get('weight').var()
        #print("Shape w = {}".format(w.shape))
        w_term_probs = F.softmax(w, axis=1) ** 4.0

        #w_topic_probs = F.softmax(w, axis=0) ** 2.0
        #print("Term 1 = {}".format(w_term_probs[0].asnumpy()))

        entropies = -F.sum(w_term_probs * F.log(w_term_probs))

        #entropies = -F.sum(w_topic_probs * F.log(w_topic_probs))
        #entropies_term = -F.sum(w_term_probs * F.log(w_term_probs), axis=1)
        #print("Shape entropies = {}".format(entropies_term.shape))        
        #print("Entropies term = {}".format(entropies_term[:20].asnumpy()))
        return (F.broadcast_add(cur_loss, entropies), entropies)


    def run_encode(self, F, in_data, batch_size):
        enc_out = self.encoder(in_data)
        #z_do = self.post_sample_dr_o(z)
        return self.latent_dist(enc_out, batch_size)

    def get_loss_terms(self, F, data, y, KL, l1_pen_const, batch_size):
        l1_pen = self.get_l1_penalty_term(F, l1_pen_const, batch_size)
        recon_loss = -F.sparse.sum( data * F.log(y+1e-12), axis=0, exclude=True )
        i_loss = F.broadcast_plus(recon_loss, F.broadcast_plus(l1_pen, KL))
        ii_loss, coherence_loss = self.add_coherence_reg_penalty(F, i_loss)
        iii_loss, entropies = self.add_seed_constraint_loss(F, ii_loss)
        #iv_loss, entropies = self.general_entropy_min_loss(F, iii_loss)
        return iii_loss, recon_loss, l1_pen, entropies, coherence_loss

    def hybrid_forward(self, F, data, l1_pen_const=None):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        emb_out = self.embedding(data)
        z, KL = self.run_encode(F, emb_out, batch_size)
        dec_out = self.decoder(z)
        y = F.softmax(dec_out, axis=1)
        iii_loss, recon_loss, l1_pen, entropies, coherence_loss = self.get_loss_terms(F, data, y, KL, l1_pen_const, batch_size)
        return iii_loss, KL, recon_loss, l1_pen, entropies, coherence_loss, y


class MetaDataBowNTM(BowNTM):

    def __init__(self, l_map, n_covars, vocabulary, enc_dim, n_latent, embedding_size,
                 fixed_embedding=False, latent_distrib='logistic_gaussian',
                 init_l1=0.0, coherence_reg_penalty=0.0, kappa=100.0, alpha=1.0, batch_size=None, n_encoding_layers=1,
                 enc_dr=0.1, wd_freqs=None, seed_mat=None, covar_net_layers=1, ctx=mx.cpu()):
        super(MetaDataBowNTM, self).__init__(vocabulary, enc_dim, n_latent, embedding_size, fixed_embedding, latent_distrib, init_l1,
                                             coherence_reg_penalty, kappa, alpha, 0.0, batch_size, n_encoding_layers, enc_dr,
                                             wd_freqs, seed_mat, n_covars, ctx)
        self.n_covars = n_covars
        self.label_map = l_map
        self.covar_net_layers = covar_net_layers
        with self.name_scope():
            if l_map is None:  
                self.cov_decoder = ContinuousCovariateModel(self.n_latent, self.n_covars, self.vocab_size,
                                                            total_layers=covar_net_layers, ctx=ctx)
            else:
                self.cov_decoder = CovariateModel(self.n_latent, self.n_covars, self.vocab_size,
                                                  batch_size=self.batch_size, interactions=True, ctx=ctx)


    def encode_data_with_covariates(self, data, covars):
        """
        Encode data to the mean of the latent distribution defined by the input `data`
        """
        emb_out = self.embedding(data)
        enc_out = self.encoder(mx.nd.concat(emb_out, covars))
        return self.latent_dist.mu_encoder(enc_out)


    def get_top_k_terms_with_covar_at_data(self, data, k, covar):
        """
        Uses test/training data-point as the input points around which term sensitivity is computed
        """
        if isinstance(covar, float):
            covar = mx.nd.array([covar], ctx=self.model_ctx)
        cv = mx.nd.expand_dims(covar, axis=0)
        jacobian = mx.nd.zeros(shape=(self.vocab_size, self.n_latent))
        for i in range(self.vocab_size):
            z_o.attach_grad()            
            with mx.autograd.record():
                #yy = self.decoder(data)
                yy = self.cov_decoder(data, cv)
                #y = yy1 + yy2
                y_i = yy[:,i] ## get y[i] across batch for i-th vocab item
            y_i.backward()
            jacobian[i] = z_o.grad
        sorted_j = jacobian.argsort(axis=0, is_ascend=False)
        return sorted_j
    

    def hybrid_forward(self, F, data, covars, l1_pen_const=None):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        emb_out = self.embedding(data)
        co_emb = F.concat(emb_out, covars)
        z, KL = self.run_encode(F, co_emb, batch_size)
        dec_out = self.decoder(z)
        cov_dec_out = self.cov_decoder(z, covars)
        y = F.softmax(dec_out + cov_dec_out, axis=1)
        iii_loss, recon_loss, l1_pen, entropies, coherence_loss = self.get_loss_terms(F, data, y, KL, l1_pen_const, batch_size)
        return iii_loss, KL, recon_loss, l1_pen, entropies, coherence_loss, y

        
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
                self.cov_inter_decoder = gluon.nn.Dense(in_units = self.n_covars * self.n_topics, units=self.vocab_size, 
                                                       activation=None, use_bias=False)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
                

    def hybrid_forward(self, F, topic_distrib, covars):
        score_C = self.cov_decoder(covars)
        if self.interactions:
            td_rsh = F.expand_dims(topic_distrib, 1)
            cov_rsh = F.expand_dims(covars, 2)
            cov_interactions = cov_rsh * td_rsh    ## shape (N, Topics, Covariates) -- outer product
            batch_size = cov_interactions.shape[0] if F is mx.ndarray else self.batch_size
            cov_interactions_rsh = F.reshape(cov_interactions, (batch_size, self.n_topics * self.n_covars))
            score_CI = self.cov_inter_decoder(cov_interactions_rsh)
            return score_CI + score_C
        else:
            return score_C
            

class ContinuousCovariateModel(HybridBlock):

    def __init__(self, n_topics, n_scalars, vocab_size, total_layers = 1, ctx=mx.cpu()):
        self.n_topics  = n_topics
        self.n_scalars = n_scalars   # number of continuous variables
        self.model_ctx = ctx
        self.time_topic_dim = 300
        super(ContinuousCovariateModel, self).__init__()

        with self.name_scope():
            self.cov_decoder = gluon.nn.HybridSequential()
            for i in range(total_layers):
                if i < 1:
                    in_units = self.n_scalars + self.n_topics
                else:
                    in_units = self.time_topic_dim
                self.cov_decoder.add(gluon.nn.Dense(in_units = in_units, units=self.time_topic_dim,
                                                    activation='relu', use_bias=(i < 1)))
            self.cov_decoder.add(gluon.nn.Dense(in_units=self.time_topic_dim, units=vocab_size, activation=None, use_bias=False))
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)

    def hybrid_forward(self, F, topic_distrib, scalars):
        inputs = F.concat(topic_distrib, scalars)
        sc_transform = self.cov_decoder(inputs)
        return sc_transform
        

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
        
