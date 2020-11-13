# coding: utf-8
"""
Copyright (c) 2019-2020. The MITRE Corporation.
"""

import mxnet as mx
from mxnet import gluon
import math
import os
import numpy as np
import gluonnlp as nlp
import logging
from mxnet.gluon.block import HybridBlock, Block

from tmnt.distribution import LogisticGaussianDistribution
from tmnt.distribution import GaussianDistribution
from tmnt.distribution import HyperSphericalDistribution
from tmnt.distribution import GaussianUnitVarDistribution


class BowVAEModel(HybridBlock):
    """
    Defines the neural architecture for a bag-of-words topic model.

    Parameters:
        vocabulary (int): Size of the vocabulary
        enc_dim (int): Number of dimension of input encoder (first FC layer)
        n_latent (int): Number of dimensions of the latent dimension (i.e. number of topics)
        embedding_size (int): Number of dimensions for embedding layer
        fixed_embedding (bool): Whether to fix embedding weights (default = False)
        latent_distrib (str): Latent distribution. 'vmf' | 'logistic_gaussian' | 'gaussian' (default = 'logistic_gaussian')
        kappa (float): Concentration parameter for vmf
        alpha (float): Hyperparameter to define prior variance for logistic gaussian
        batch_size (int): provided only at training time (or when model is Hybridized) - otherwise will be inferred (default None) 
        n_encoding_layers (int): Number of layers used for the encoder. (default = 1)
        enc_dr (float): Dropout after each encoder layer. (default = 0.1)
        wd_freqs (:class:`mxnet.ndarray.NDArray`): Tensor with word frequencies in training data to initialize bias terms.
        seed_mat (:class:`mxnet.ndarray.NDArray`): Tensor with seed terms for guided topic modeling loss
        n_covars (int): Number of values for categorical co-variate (0 for non-MetaData BOW model)
        ctx (int): context device (default is mx.cpu())
    """
    def __init__(self, vocabulary, enc_dim, n_latent, embedding_size, fixed_embedding=False, latent_distrib='logistic_gaussian',
                 coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0,
                 kappa=100.0, alpha=1.0, batch_size=None, n_encoding_layers = 1, enc_dr=0.1,
                 wd_freqs=None, seed_mat=None, n_covars=0, ctx=mx.cpu()):
        super(BowVAEModel, self).__init__()
        self.batch_size = batch_size
        self._orig_batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.vocab_size = len(vocabulary)
        self.coherence_reg_penalty = coherence_reg_penalty
        self.redundancy_reg_penalty = redundancy_reg_penalty
        self.embedding_size = embedding_size
        self.vocabulary = vocabulary
        self.num_enc_layers = n_encoding_layers
        if vocabulary.embedding:
            assert vocabulary.embedding.idx_to_vec[0].size == embedding_size
        self.encoding_dims = [self.embedding_size + n_covars] + [enc_dim for _ in range(n_encoding_layers)]
        with self.name_scope():
            ## Add in topic seed constraints
            self.seed_matrix = seed_mat
            ## should be tanh here to avoid losing embedding information
            self.embedding = gluon.nn.Dense(in_units=self.vocab_size, units=self.embedding_size, activation='tanh')
            self.encoder = self._get_encoder(self.encoding_dims, dr=enc_dr)
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianDistribution(n_latent, ctx, alpha=alpha)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalDistribution(n_latent, kappa=kappa, ctx=self.model_ctx)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianDistribution(n_latent, ctx)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarDistribution(n_latent, ctx)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.vocab_size, activation=None)
            self.coherence_regularization = CoherenceRegularizer(self.coherence_reg_penalty, self.redundancy_reg_penalty)
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        ## vmf needs to set weight values post-initialization
        self.latent_dist.post_init(self.model_ctx)
        if vocabulary.embedding:            
            emb = vocabulary.embedding.idx_to_vec.transpose()
            emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=0) + 1e-10
            emb_norm = emb / emb_norm_val
            self.embedding.weight.set_data(emb_norm)
            if fixed_embedding:
                self.embedding.collect_params().setattr('grad_req', 'null')
        ## Initialize and FIX decoder bias terms to corpus frequencies
        if wd_freqs is not None:
            self.set_biases(wd_freqs)

    def set_biases(self, wd_freqs):
        """Set the biases to the log of the word frequencies.

        Parameters:
            wd_freqs (:class:`mxnet.ndarray.NDArray`): Word frequencies as determined from training data
        """
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

    def get_ordered_terms_encoder(self, dataloader, sample_size=-1):
        jacobians = np.zeros(shape=(self.n_latent, self.vocab_size))
        samples = 0
        for bi, (data, _) in enumerate(dataloader):
            if sample_size > 0 and samples >= sample_size:
                print("Sample processed, exiting..")
                break
            samples += data.shape[0]
            x_data = data.tostype('default')
            x_data = x_data.as_in_context(self.model_ctx)
            for i in range(self.n_latent):
                x_data.attach_grad()
                with mx.autograd.record():
                    emb_out = self.embedding(x_data)
                    enc_out = self.latent_dist.mu_encoder(self.encoder(emb_out))
                    yi = enc_out[:, i] ## for the ith topic, over batch
                yi.backward()
                mx.nd.waitall()
                ss = x_data.grad.sum(axis=0).asnumpy()
                jacobians[i] += ss
        sorted_j = (- jacobians).argsort(axis=1).transpose()
        return sorted_j

    def get_ordered_terms(self):
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
        return sorted_j.asnumpy()
    

    def get_topic_vectors(self):
        """
        Returns unnormalized topic vectors
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
        return jacobian.asnumpy()        


    def encode_data(self, data):
        """
        Encode data to the mean of the latent distribution defined by the input `data`.

        Parameters
        ----------
        data: `mxnet.ndarray.NDArray` or `mxnet.symbol.Symbol` 
            input data of shape (batch_size, vocab_size)

        Returns
        -------
        `mxnet.ndarray.NDArray` or `mxnet.symbol.Symbol`
            Result of encoding with shape (batch_size, n_latent)
        """
        return self.latent_dist.mu_encoder(self.encoder(self.embedding(data)))
    
    def add_coherence_reg_penalty(self, F, cur_loss):
        if self.coherence_reg_penalty > 0.0:
            if F is mx.ndarray:
                w = self.decoder.params.get('weight').data()
                emb = self.embedding.params.get('weight').data()
            else:
                w = self.decoder.params.get('weight').var()
                emb = self.embedding.params.get('weight').var()
            c, d = self.coherence_regularization(w, emb)
            return (cur_loss + c + d), c, d
        else:
            return (cur_loss, F.zeros_like(cur_loss), F.zeros_like(cur_loss))

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

    def run_encode(self, F, in_data, batch_size):
        enc_out = self.encoder(in_data)
        return self.latent_dist(enc_out, batch_size)

    def get_loss_terms(self, F, data, y, KL, batch_size):
        rr = data * F.log(y+1e-12)
        recon_loss = -F.sparse.sum( rr, axis=1 )
        i_loss = F.broadcast_plus(recon_loss, KL)
        ii_loss, coherence_loss, redundancy_loss = self.add_coherence_reg_penalty(F, i_loss)
        iii_loss, entropies = self.add_seed_constraint_loss(F, ii_loss)
        return iii_loss, recon_loss, entropies, coherence_loss, redundancy_loss

    def hybrid_forward(self, F, data):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        emb_out = self.embedding(data)
        z, KL = self.run_encode(F, emb_out, batch_size)
        dec_out = self.decoder(z)
        y = F.softmax(dec_out, axis=1)
        iii_loss, recon_loss, entropies, coherence_loss, redundancy_loss = \
            self.get_loss_terms(F, data, y, KL, batch_size)
        return iii_loss, KL, recon_loss, entropies, coherence_loss, redundancy_loss, y


class MetaDataBowVAEModel(BowVAEModel):

    def __init__(self, l_map, n_covars, vocabulary, enc_dim, n_latent, embedding_size,
                 fixed_embedding=False, latent_distrib='logistic_gaussian',
                 coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0, kappa=100.0, alpha=1.0,
                 batch_size=None, n_encoding_layers=1,
                 enc_dr=0.1, wd_freqs=None, seed_mat=None, covar_net_layers=1, ctx=mx.cpu()):
        super(MetaDataBowVAEModel, self).__init__(vocabulary, enc_dim, n_latent, embedding_size, fixed_embedding,
                                             latent_distrib, 
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



    def get_ordered_terms_with_covar_at_data(self, data, k, covar):
        """
        Uses test/training data-point as the input points around which term sensitivity is computed
        """
        data = data.as_in_context(self.model_ctx)
        covar = covar.as_in_context(self.model_ctx)
        jacobian = mx.nd.zeros(shape=(self.vocab_size, self.n_latent), ctx=self.model_ctx)

        batch_size = data.shape[0]
        emb_out = self.embedding(data)

        co_emb = mx.nd.concat(emb_out, covar)
        z = self.latent_dist.mu_encoder(self.encoder(co_emb))

        z.attach_grad()
        outputs = []
        with mx.autograd.record():

            dec_out = self.decoder(z)
            cov_dec_out = self.cov_decoder(z, covar)

            y = mx.nd.softmax(cov_dec_out + dec_out, axis=1)
            for i in range(self.vocab_size):
                outputs.append(y[:,i])

        for i, output in enumerate(outputs):
            output.backward(retain_graph=True)
            jacobian[i] += z.grad.sum(axis=0)
        sorted_j = jacobian.argsort(axis=0, is_ascend=False)
        return sorted_j

    def get_topic_vectors(self, data, covar):
        """
        Returns unnormalized topic vectors based on the input data
        """
        data = data.as_in_context(self.model_ctx)
        covar = covar.as_in_context(self.model_ctx)
        jacobian = mx.nd.zeros(shape=(self.vocab_size, self.n_latent), ctx=self.model_ctx)

        batch_size = data.shape[0]
        emb_out = self.embedding(data)

        co_emb = mx.nd.concat(emb_out, covar)
        z = self.latent_dist.mu_encoder(self.encoder(co_emb))

        z.attach_grad()
        outputs = []
        with mx.autograd.record():

            dec_out = self.decoder(z)
            cov_dec_out = self.cov_decoder(z, covar)

            y = mx.nd.softmax(cov_dec_out + dec_out, axis=1)
            for i in range(self.vocab_size):
                outputs.append(y[:,i])

        for i, output in enumerate(outputs):
            output.backward(retain_graph=True)
            jacobian[i] += z.grad.sum(axis=0)
        return jacobian
        


    def hybrid_forward(self, F, data, covars):
        batch_size = data.shape[0] if F is mx.ndarray else self.batch_size
        emb_out = self.embedding(data)
        co_emb = F.concat(emb_out, covars)
        z, KL = self.run_encode(F, co_emb, batch_size)
        dec_out = self.decoder(z)
        cov_dec_out = self.cov_decoder(z, covars)
        y = F.softmax(dec_out + cov_dec_out, axis=1)
        iii_loss, recon_loss, entropies, coherence_loss, redundancy_loss = \
            self.get_loss_terms(F, data, y, KL, batch_size)
        return iii_loss, KL, recon_loss, entropies, coherence_loss, redundancy_loss, y

        
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

    ## Follows paper to add coherence loss: http://aclweb.org/anthology/D18-1096

    def __init__(self, coherence_pen=1.0, redundancy_pen=1.0):
        super(CoherenceRegularizer, self).__init__()
        self.coherence_pen = coherence_pen
        self.redundancy_pen = redundancy_pen
        

    def hybrid_forward(self, F, w, emb):
        ## emb should have shape (D x V)
        ## w should have shape (V x K)
        # w NORM over columns
        w_min = F.min(w, keepdims=True, axis=0)
        ww = w - w_min # ensure weights are non-negative
        w_norm_val = F.norm(ww, keepdims=True, axis=0)
        emb_norm_val = F.norm(emb, keepdims=True, axis=1)
        
        w_norm = F.broadcast_div(ww, w_norm_val)
        emb_norm = F.broadcast_div(emb, emb_norm_val)

        T = F.linalg.gemm2(emb_norm, w_norm)
        T_norm_vals = F.norm(T, keepdims=True, axis=0)
        T_norm = F.broadcast_div(T, T_norm_vals)

        S = F.linalg.gemm2(F.transpose(emb_norm), T_norm) # (V x K)
        C = -F.sum(S * w_norm)
        ## diversity component
        D1 = F.linalg.gemm2(F.transpose(T_norm), T_norm)
        D = F.sum(D1)
        return C * self.coherence_pen , D * self.redundancy_pen



class BertBowVED(Block):
    def __init__(self, bert_base, bow_vocab, latent_distrib='vmf', 
                 n_latent=256, 
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_freqs=None,
                 redundancy_reg_penalty=0.0,
                 ctx = mx.cpu(),
                 prefix=None, params=None):
        super(BertBowVED, self).__init__(prefix=prefix, params=params)
        self.kld_wt = kld
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.batch_size = batch_size
        self.bow_vocab_size = len(bow_vocab)
        self.vocabulary = bow_vocab
        self.latent_distrib = latent_distrib
        self.kappa = kappa
        self.coherence_reg_penalty = 0.0
        self.redundancy_reg_penalty = redundancy_reg_penalty
        with self.name_scope():
            self.encoder = bert_base            
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalDistribution(n_latent, kappa=kappa, ctx=self.model_ctx, dr=0.0)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianDistribution(n_latent, ctx, dr=0.0)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarDistribution(n_latent, ctx, dr=0.0, var=0.05)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=self.bow_vocab_size, activation=None)
            self.decoder.initialize(mx.init.Xavier(), ctx=self.model_ctx)
            self.latent_dist.initialize(mx.init.Xavier(), ctx=self.model_ctx)
            self.latent_dist.post_init(self.model_ctx)
            self.coherence_regularization = CoherenceRegularizer(self.coherence_reg_penalty, self.redundancy_reg_penalty)
            if self.vocabulary.embedding:
                self.embedding = gluon.nn.Dense(in_units=len(self.vocabulary), units = self.vocabulary.embedding.idx_to_vec[0].size, use_bias=False)
                self.embedding.initialize(mx.init.Xavier(), ctx=self.model_ctx)
                emb = self.vocabulary.embedding.idx_to_vec.transpose()
                emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=0) + 1e-10
                emb_norm = emb / emb_norm_val
                self.embedding.collect_params().setattr('grad_req', 'null')
                                                
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1
            total = freq_nd.sum()
            log_freq = freq_nd.log() - freq_nd.sum().log()
            bias_param = self.decoder.collect_params().get('bias')
            bias_param.set_data(log_freq)
            bias_param.grad_req = 'null'
            self.out_bias = bias_param.data()

    def get_top_k_terms(self, k):
        """
        Returns the top K terms for each topic based on sensitivity analysis. Terms whose 
        probability increases the most for a unit increase in a given topic score/probability
        are those most associated with the topic. This is just the topic-term weights for a 
        linear decoder - but code here will work with arbitrary decoder.
        """
        z = mx.nd.ones(shape=(1, self.n_latent), ctx=self.model_ctx)
        jacobian = mx.nd.zeros(shape=(self.bow_vocab_size, self.n_latent), ctx=self.model_ctx)
        z.attach_grad()        
        for i in range(self.bow_vocab_size):
            with mx.autograd.record():
                y = self.decoder(z)
                yi = y[0][i]
            yi.backward()
            jacobian[i] = z.grad
        sorted_j = jacobian.argsort(axis=0, is_ascend=False)
        return sorted_j.asnumpy()

    def __call__(self, toks, tok_types, valid_length, bow):
        return super(BertBowVED, self).__call__(toks, tok_types, valid_length, bow)

    def set_kl_weight(self, epoch, max_epochs):
        burn_in = int(max_epochs / 10)
        eps = 1e-6
        if epoch > burn_in:
            self.kld_wt = ((epoch - burn_in) / (max_epochs - burn_in)) + eps
        else:
            self.kld_wt = eps
        return self.kld_wt

    def add_coherence_reg_penalty(self, cur_loss):
        if (self.coherence_reg_penalty > 0.0  or self.redundancy_reg_penalty > 0) and self.embedding is not None:
            w = self.decoder.params.get('weight').data()
            emb = self.embedding.params.get('weight').data()
            _, redundancy_loss = self.coherence_regularization(w, emb)
            return (cur_loss + redundancy_loss), redundancy_loss
        else:
            return cur_loss, mx.nd.zeros_like(cur_loss)

    def encode(self, toks, tok_types, valid_length):
        _, enc = self.encoder(toks, tok_types, valid_length)
        return self.latent_dist.mu_encoder(enc)

    def forward(self, toks, tok_types, valid_length, bow):
        _, enc = self.encoder(toks, tok_types, valid_length)
        z, KL = self.latent_dist(enc, self.batch_size)
        y = self.decoder(z)
        y = mx.nd.softmax(y, axis=1)
        rr = bow * mx.nd.log(y+1e-12)
        recon_loss = -mx.nd.sparse.sum( rr, axis=1 )
        KL_loss = ( KL * self.kld_wt )
        loss = recon_loss + KL_loss
        ii_loss, redundancy_loss = self.add_coherence_reg_penalty(loss)
        return ii_loss, recon_loss, KL_loss, redundancy_loss, y

    def get_encoder_jacobian(dataloader, batch_size, sample_size):
        jacobians = np.zeros(shape=(model.n_latent, model.vocab_size))        
        for bi, seqs in enumerate(dataloader):
            if bi * batch_size >= sample_size:
                print("Sample processed, exiting..")
                break
            input_ids, valid_length, type_ids, _ = seqs
            input_ids_x = input_ids.as_in_context(self.model_ctx)
            valid_length_x = input_ids.as_in_context(self.model_ctx)
            type_ids_x = input_ids.as_in_context(self.model_ctx)
            for i in range(model.n_latent):
                x_data.attach_grad()
                with mx.autograd.record():
                    _, enc = self.encoder(input_ids_x, type_ids_x, valid_length_x)
                    enc_out = model.latent_dist.mu_encoder(enc)
                    yi = enc_out[:, i] ## for the ith topic, over batch
                yi.backward()
                mx.nd.waitall()
                ss = x_data.grad.sum(axis=0).asnumpy()
                jacobians[i] += ss
        return jacobians
    
