# coding: utf-8
# Copyright (c) 2019-2021. The MITRE Corporation.
"""
Core Neural Net architectures for topic modeling.
"""

import mxnet as mx
from mxnet import gluon
import math
import os
import numpy as np
import gluonnlp as nlp
import logging
from mxnet.gluon import HybridBlock, Block
from mxnet.gluon import nn

from tmnt.distribution import LogisticGaussianDistribution
from tmnt.distribution import GaussianDistribution
from tmnt.distribution import HyperSphericalDistribution
from tmnt.distribution import GaussianUnitVarDistribution
from mxnet.gluon.loss import Loss, KLDivLoss

class BaseVAE(HybridBlock):

    def __init__(self, vocabulary=None, latent_distribution=LogisticGaussianDistribution(20),
                 coherence_reg_penalty=0.0, redundancy_reg_penalty=0.0,
                 n_covars=0, ctx=mx.cpu(), **kwargs):
        super(BaseVAE, self).__init__(**kwargs)        
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)        
        self.n_latent   = latent_distribution.n_latent
        self.coherence_reg_penalty = coherence_reg_penalty
        self.redundancy_reg_penalty = redundancy_reg_penalty
        self.n_covars = n_covars
        self.model_ctx = ctx
        self.embedding = None

        ## common aspects of all(most!) variational topic models
        with self.name_scope():
            self.latent_distribution = latent_distribution
            self.decoder = gluon.nn.Dense(in_units=self.n_latent, units=self.vocab_size, activation=None)
            self.coherence_regularization = CoherenceRegularizer(self.coherence_reg_penalty, self.redundancy_reg_penalty)

    def initialize_bias_terms(self, wd_freqs):
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1 # simple smoothing
            total = freq_nd.sum()
            log_freq = freq_nd.log() - freq_nd.sum().log()
            bias_param = self.decoder.collect_params().get('bias')
            bias_param.set_data(log_freq)
            bias_param.grad_req = 'null'
            self.out_bias = bias_param.data()            

    def get_ordered_terms(self):
        """
        Returns the top K terms for each topic based on sensitivity analysis. Terms whose 
        probability increases the most for a unit increase in a given topic score/probability
        are those most associated with the topic.
        """
        z = mx.nd.ones(shape=(1, self.n_latent), ctx=self.model_ctx)
        jacobian = mx.nd.zeros(shape=(self.vocab_size, self.n_latent), ctx=self.model_ctx)
        for i in range(self.vocab_size):
            z.attach_grad()        
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


    def add_coherence_reg_penalty(self, F, cur_loss):
        if self.coherence_reg_penalty > 0.0 and self.embedding is not None:
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

    def get_loss_terms(self, F, data, y, KL, batch_size):
        rr = data * F.log(y+1e-12)
        recon_loss = -F.sparse.sum( rr, axis=1 )
        i_loss = F.broadcast_plus(recon_loss, KL)
        ii_loss, coherence_loss, redundancy_loss = self.add_coherence_reg_penalty(F, i_loss)
        return ii_loss, recon_loss, coherence_loss, redundancy_loss




class BowVAEModel(BaseVAE):
    """
    Defines the neural architecture for a bag-of-words topic model.

    Parameters:
        vocabulary (:class:`gluon.Vocab`): GluonNLP Vocabulary
        enc_dim (int): Number of dimension of input encoder (first FC layer)
        embedding_size (int): Number of dimensions for embedding layer
        fixed_embedding (bool): Whether to fix embedding weights (default = False)
        n_encoding_layers (int): Number of layers used for the encoder. (default = 1)
        enc_dr (float): Dropout after each encoder layer. (default = 0.1)
        n_covars (int): Number of values for categorical co-variate (0 for non-CovariateData BOW model)
        ctx (int): context device (default is mx.cpu())
    """
    def __init__(self,
                 enc_dim, embedding_size, n_encoding_layers, enc_dr, fixed_embedding,
                 n_labels=0,
                 gamma=1.0,
                 multilabel=False,
                 classifier_dropout=0.1,
                 *args, **kwargs):
        super(BowVAEModel, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.num_enc_layers = n_encoding_layers
        self.enc_dr = enc_dr
        self.enc_dim = enc_dim
        self.multilabel = multilabel
        self.n_labels = n_labels
        self.gamma    = gamma
        self.classifier_dropout=classifier_dropout
        self.has_classifier = self.n_labels > 1
        if self.vocabulary.embedding:
            assert self.vocabulary.embedding.idx_to_vec[0].size == self.embedding_size
        self.encoding_dims = [self.embedding_size + self.n_covars] + [enc_dim for _ in range(n_encoding_layers)]
        
        with self.name_scope():
            self.embedding = gluon.nn.Dense(in_units=self.vocab_size, units=self.embedding_size, activation='tanh')
            self.encoder = self._get_encoder(self.encoding_dims, dr=enc_dr)
            if self.has_classifier:
                self.lab_dr = gluon.nn.Dropout(self.enc_dr*2.0)
                self.classifier = gluon.nn.Dense(in_units=self.n_latent, units=self.n_labels, activation=None, use_bias=True)
            #self.lab_loss_fn = gluon.loss.SigmoidBCELoss() if multilabel else gluon.loss.SoftmaxCELoss()
            
        self.initialize(mx.init.Xavier(), ctx=self.model_ctx)
        ## vmf needs to set weight values post-initialization
        self.latent_distribution.post_init(self.model_ctx)
        if self.vocabulary.embedding:            
            emb = self.vocabulary.embedding.idx_to_vec.transpose()
            emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=0) + 1e-10
            emb_norm = emb / emb_norm_val
            self.embedding.weight.set_data(emb_norm)
            if fixed_embedding:
                self.embedding.collect_params().setattr('grad_req', 'null')

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
            x_data = mx.nd.minimum(x_data, 1.0)
            for i in range(self.n_latent):
                x_data.attach_grad()
                with mx.autograd.record():
                    emb_out = self.embedding(x_data)
                    enc_out = self.latent_distribution.get_mu_encoding(self.encoder(emb_out), include_bn=True)
                    yi = enc_out[:, i] ## for the ith topic, over batch
                dx = mx.autograd.grad(yi, x_data, train_mode=False)
                ss = dx[0].sum(axis=0).asnumpy()
                jacobians[i] += ss
        sorted_j = (- jacobians).argsort(axis=1).transpose()
        return sorted_j

    def get_ordered_terms_per_item(self, dataloader, sample_size=-1):
        jacobian_list = [[] for i in range(self.n_latent)]
        samples = 0
        for bi, (data, _) in enumerate(dataloader):
            if sample_size > 0 and samples >= sample_size:
                print("Sample processed, exiting..")
                break
            samples += data.shape[0]
            x_data = data.tostype('default')
            x_data = x_data.as_in_context(self.model_ctx)
            x_data = mx.nd.minimum(x_data, 1.0)
            for i in range(self.n_latent):
                x_data.attach_grad()
                with mx.autograd.record():
                    emb_out = self.embedding(x_data)
                    enc_out = self.latent_distribution.get_mu_encoding(self.encoder(emb_out), include_bn=True)
                    yi = enc_out[:, i] ## for the ith topic, over batch
                dx = mx.autograd.grad(yi, x_data, train_mode=False)
                ss = dx[0].asnumpy()
                jacobian_list[i] += list(ss)
        return jacobian_list


    def encode_data(self, data, include_bn=True):
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
        return self.latent_distribution.get_mu_encoding(self.encoder(self.embedding(data)), include_bn=include_bn)
    

    def run_encode(self, F, in_data, batch_size):
        enc_out = self.encoder(in_data)
        return self.latent_distribution(enc_out, batch_size)


    def predict(self, data):
        """Predict the label given the input data (ignoring VAE reconstruction)
        
        Parameters:
            data (tensor): input data tensor
        Returns:
            output vector (tensor): unnormalized outputs over label values
        """
        return self.classifier(self.lab_dr(self.encode_data(data)))
    

    def hybrid_forward(self, F, data):
        batch_size = data.shape[0]
        emb_out = self.embedding(data)
        #z, KL = self.run_encode(F, emb_out, batch_size)
        enc_out = self.encoder(emb_out)
        mu_out  = self.latent_distribution.get_mu_encoding(enc_out)
        z, KL   = self.latent_distribution(enc_out, batch_size)
        y = F.softmax(self.decoder(z), axis=1)
        ii_loss, recon_loss, coherence_loss, redundancy_loss = \
            self.get_loss_terms(F, data, y, KL, batch_size)
        if self.has_classifier:
            classifier_outputs = self.classifier(self.lab_dr(mu_out))
        else:
            classifier_outputs = None
        return ii_loss, KL, recon_loss, coherence_loss, redundancy_loss, classifier_outputs


class MetricBowVAEModel(BowVAEModel):

    def __init__(self, *args, **kwargs):
        self.kld_wt = 1.0
        super(MetricBowVAEModel, self).__init__(*args, **kwargs)


    def get_redundancy_penalty(self):
        w = self.decoder.params.get('weight').data()
        emb = self.embedding.params.get('weight').data() if self.embedding is not None else w.transpose()
        _, redundancy_loss = self.coherence_regularization(w, emb)
        return redundancy_loss
        

    def _get_elbo(self, bow, enc):
        batch_size = bow.shape[0]
        z, KL = self.latent_distribution(enc, batch_size)
        KL_loss = (KL * self.kld_wt)
        y = mx.nd.softmax(self.decoder(z), axis=1)
        rec_loss = -mx.nd.sum( bow * mx.nd.log(y+1e-12), axis=1 )
        elbo = rec_loss + KL_loss
        return elbo, rec_loss, KL_loss

    def _get_encoding(self, data):
        return self.encoder( self.embedding(data) )

    def unpaired_input_forward(self, data):
        enc = self._get_encoding(data)
        elbo, rec_loss, kl_loss = self._get_elbo(data, enc)
        redundancy_loss = self.get_redundancy_penalty()
        return elbo, rec_loss, kl_loss, redundancy_loss


    def hybrid_forward(self, F, data1, data2):
        enc1 = self._get_encoding(data1)
        enc2 = self._get_encoding(data2)
        mu1  = self.latent_distribution.get_mu_encoding(enc1)
        mu2  = self.latent_distribution.get_mu_encoding(enc2)
        elbo1, rec_loss1, KL_loss1 = self._get_elbo(data1, enc1)
        elbo2, rec_loss2, KL_loss2 = self._get_elbo(data2, enc2)        
        redundancy_loss = self.get_redundancy_penalty()
        return (elbo1 + elbo2), (rec_loss1 + rec_loss2), (KL_loss1 + KL_loss2), redundancy_loss, mu1, mu2


class CovariateBowVAEModel(BowVAEModel):
    """Bag-of-words topic model with labels used as co-variates
    """
    def __init__(self, covar_net_layers=1, *args, **kwargs):
        super(CovariateBowVAEModel, self).__init__(*args, **kwargs)
        self.covar_net_layers = covar_net_layers
        with self.name_scope():
            if self.n_covars < 1:  
                self.cov_decoder = ContinuousCovariateModel(self.n_latent, self.vocab_size,
                                                            total_layers=self.covar_net_layers, ctx=self.model_ctx)
            else:
                self.cov_decoder = CovariateModel(self.n_latent, self.n_covars, self.vocab_size,
                                                  interactions=True, ctx=self.model_ctx)


    def encode_data_with_covariates(self, data, covars, include_bn=False):
        """
        Encode data to the mean of the latent distribution defined by the input `data`
        """
        emb_out = self.embedding(data)
        enc_out = self.encoder(mx.nd.concat(emb_out, covars))
        return self.latent_distribution.get_mu_encoding(enc_out, include_bn=include_bn)



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
        z = self.latent_distribution.get_mu_encoding(self.encoder(co_emb))

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
        z = self.latent_distribution.get_mu_encoding(self.encoder(co_emb))
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
        batch_size = data.shape[0]
        emb_out = self.embedding(data)
        if self.n_covars > 0:
            covars = F.one_hot(covars, self.n_covars)
        co_emb = F.concat(emb_out, covars)
        z, KL = self.run_encode(F, co_emb, batch_size)
        dec_out = self.decoder(z)
        cov_dec_out = self.cov_decoder(z, covars)
        y = F.softmax(dec_out + cov_dec_out, axis=1)
        ii_loss, recon_loss, coherence_loss, redundancy_loss = \
            self.get_loss_terms(F, data, y, KL, batch_size)
        return ii_loss, KL, recon_loss, coherence_loss, redundancy_loss, None

        
class CovariateModel(HybridBlock):

    def __init__(self, n_topics, n_covars, vocab_size, interactions=False, ctx=mx.cpu()):
        self.n_topics = n_topics
        self.n_covars = n_covars
        self.vocab_size = vocab_size
        self.interactions = interactions
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
            batch_size = cov_interactions.shape[0]
            cov_interactions_rsh = F.reshape(cov_interactions, (batch_size, self.n_topics * self.n_covars))
            score_CI = self.cov_inter_decoder(cov_interactions_rsh)
            return score_CI + score_C
        else:
            return score_C
            

class ContinuousCovariateModel(HybridBlock):

    def __init__(self, n_topics, vocab_size, total_layers = 1, ctx=mx.cpu()):
        self.n_topics  = n_topics
        self.n_scalars = 1   # number of continuous variables
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
        T_norm = F.broadcast_div(T, T_norm_vals) # (D x K)

        S = F.linalg.gemm2(F.transpose(emb_norm), T_norm) # (V x K)
        C = -F.sum(S * w_norm)
        ## diversity component
        D1 = F.linalg.gemm2(F.transpose(T_norm), T_norm)
        D = F.sum(D1)
        return C * self.coherence_pen , D * self.redundancy_pen



class BaseSeqBowVED(Block):
    def __init__(self,
                 bert,
                 latent_dist,
                 num_classes=0,
                 dropout=0.0,
                 bow_vocab_size=2000,
                 n_latent=20, 
                 kld=0.1,
                 ctx=mx.cpu(),
                 redundancy_reg_penalty=0.0, pre_trained_embedding = None):
        super(BaseSeqBowVED, self).__init__()
        self.n_latent = latent_dist.n_latent
        self.bert = bert
        self.kld_wt = kld
        self.has_classifier = num_classes >= 2
        self.num_classes = num_classes
        self.dropout = dropout
        self.bow_vocab_size = bow_vocab_size
        self.redundancy_reg_penalty = redundancy_reg_penalty
        self.vocabulary = None ### XXX - add this as option to be passed in
        self.model_ctx = ctx
        with self.name_scope():
            self.latent_dist = latent_dist
            self.embedding = None
            self.decoder = nn.Dense(in_units=self.n_latent, units=bow_vocab_size, use_bias=True)
            self.coherence_regularization = CoherenceRegularizer(0.0, self.redundancy_reg_penalty)            
            if pre_trained_embedding is not None:
                self.embedding = nn.Dense(in_units = len(pre_trained_embedding.idx_to_vec),
                                          units = pre_trained_embedding.idx_to_vec[0].size, use_bias=False)
            if self.vocabulary:
                self.embedding = gluon.nn.Dense(in_units=len(self.vocabulary),
                                                units = self.vocabulary.embedding.idx_to_vec[0].size, use_bias=False)
                self.embedding.initialize(mx.init.Xavier(), ctx=self.model_ctx)
                emb = self.vocabulary.embedding.idx_to_vec.transpose()
                emb_norm_val = mx.nd.norm(emb, keepdims=True, axis=0) + 1e-10
                emb_norm = emb / emb_norm_val
                self.embedding.collect_params().setattr('grad_req', 'null')

    def get_redundancy_penalty(self):
        w = self.decoder.params.get('weight').data()
        emb = self.embedding.params.get('weight').data() if self.embedding is not None else w.transpose()
        _, redundancy_loss = self.coherence_regularization(w, emb)
        return redundancy_loss


    def initialize_bias_terms(self, wd_freqs):
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1 # simple smoothing
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
            

class SeqBowVED(BaseSeqBowVED):
    def __init__(self, *args, **kwargs):
        super(SeqBowVED, self).__init__(*args, **kwargs)
        with self.name_scope():
            if self.has_classifier:
                self.classifier = nn.HybridSequential()
                if self.dropout:
                    self.classifier.add(nn.Dropout(rate=self.dropout))
                self.classifier.add(nn.Dense(in_units=self.n_latent, units=self.num_classes))

    def forward(self, inputs, token_types, valid_length=None, bow=None):  # pylint: disable=arguments-differ
        _, enc = self.bert(inputs, token_types, valid_length)
        return self.forward_with_cached_encoding(inputs, enc, bow)

    def forward_with_cached_encoding(self, inputs, enc, bow):
        elbo, rec_loss, KL_loss = 0.0, 0.0, 0.0
        if bow is not None:
            bow = bow.squeeze(axis=1)
            z, KL = self.latent_dist(enc, inputs.shape[0])
            KL_loss = (KL * self.kld_wt)
            y = mx.nd.softmax(self.decoder(z), axis=1)
            rec_loss = -mx.nd.sum( bow * mx.nd.log(y+1e-12), axis=1 )
            elbo = rec_loss + KL_loss
        if self.has_classifier:
            z_mu = self.latent_dist.get_mu_encoding(enc)            
            classifier_outputs = self.classifier(z_mu)
        else:
            classifier_outputs = None
        redundancy_loss = self.get_redundancy_penalty()
        elbo = elbo + redundancy_loss
        return elbo, rec_loss, KL_loss, redundancy_loss, classifier_outputs


class MetricSeqBowVED(BaseSeqBowVED):
    def __init__(self, *args, **kwargs):
        super(MetricSeqBowVED, self).__init__(*args, **kwargs)

    def _get_elbo(self, bow, enc):
        bow = bow.squeeze(axis=1)
        z, KL = self.latent_dist(enc, bow.shape[0])
        KL_loss = (KL * self.kld_wt)
        y = mx.nd.softmax(self.decoder(z), axis=1)
        rec_loss = -mx.nd.sum( bow * mx.nd.log(y+1e-12), axis=1 )
        elbo = rec_loss + KL_loss
        return elbo, rec_loss, KL_loss

    def unpaired_input_forward(self, in1, tt1, vl1, bow1):
        _, enc1 = self.bert(in1, tt1, vl1)
        elbo1, rec_loss1, KL_loss1 = self._get_elbo(bow1, enc1)
        redundancy_loss = self.get_redundancy_penalty()
        return elbo1, rec_loss1, KL_loss1, redundancy_loss

    def forward(self, in1, tt1, vl1, bow1, in2, tt2, vl2, bow2):
        _, enc1 = self.bert(in1, tt1, vl1)
        _, enc2 = self.bert(in2, tt2, vl2)
        elbo1, rec_loss1, KL_loss1 = self._get_elbo(bow1, enc1)
        elbo2, rec_loss2, KL_loss2 = self._get_elbo(bow2, enc2)
        elbo = elbo1 + elbo2
        rec_loss = rec_loss1 + rec_loss2
        KL_loss = KL_loss1 + KL_loss2
        z_mu1 = self.latent_dist.get_mu_encoding(enc1)
        z_mu2 = self.latent_dist.get_mu_encoding(enc2)
        redundancy_loss = self.get_redundancy_penalty()
        return elbo, rec_loss, KL_loss, redundancy_loss, z_mu1, z_mu2


class GeneralizedSDMLLoss(Loss):
    r"""Calculates Batchwise Smoothed Deep Metric Learning (SDML) Loss given two input tensors and a smoothing weight
    SDM Loss learns similarity between paired samples by using unpaired samples in the minibatch
    as potential negative examples.

    The loss is described in greater detail in
    "Large Scale Question Paraphrase Retrieval with Smoothed Deep Metric Learning."
    - by Bonadiman, Daniele, Anjishnu Kumar, and Arpit Mittal.  arXiv preprint arXiv:1905.12786 (2019).
    URL: https://arxiv.org/pdf/1905.12786.pdf

    Parameters
    ----------
    smoothing_parameter : float
        Probability mass to be distributed over the minibatch. Must be < 1.0.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.

    Inputs:
        - **x1**: Minibatch of data points with shape (batch_size, vector_dim)
        - **x2**: Minibatch of data points with shape (batch_size, vector_dim)
          Each item in x1 is a positive sample for the items with the same label in x2
          That is, x1[0] and x2[0] form a positive pair iff label(x1[0]) = label(x2[0])
          All data points in different rows should be decorrelated

    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, smoothing_parameter=0.3, weight=1., batch_axis=0, x2_downweight_idx=-1, **kwargs):
        super(GeneralizedSDMLLoss, self).__init__(weight, batch_axis, **kwargs)
        self.kl_loss = KLDivLoss(from_logits=True)
        self.smoothing_parameter = smoothing_parameter # Smoothing probability mass
        self.x2_downweight_idx = x2_downweight_idx

    def _compute_distances(self, x1, x2):
        """
        This function computes the euclidean distance between every vector
        in the two batches in input.
        """

        # extracting sizes expecting [batch_size, dim]
        assert x1.shape == x2.shape
        batch_size, dim = x1.shape
        # expanding both tensor form [batch_size, dim] to [batch_size, batch_size, dim]
        x1_ = x1.expand_dims(1).broadcast_to([batch_size, batch_size, dim])
        x2_ = x2.expand_dims(0).broadcast_to([batch_size, batch_size, dim])
        # pointwise squared differences
        squared_diffs = (x1_ - x2_)**2
        # sum of squared differences distance
        return squared_diffs.sum(axis=2)


    def _compute_labels(self, F, l1, l2):
        """
        Example:
        l1 = [1,2,2]
        l2 = [1,2,1]
        ===> 
        [ [ 1, 0, 1],
          [ 0, 1, 0],
          [ 0, 1, 0] ]
        
        This is an outer product with the equality predicate.
        """
        l1 = l1.squeeze()
        l2 = l2.squeeze()
        batch_size = l1.shape[0]
        l1_x = F.broadcast_to(F.expand_dims(l1, 1), (batch_size, batch_size))
        l2_x = F.broadcast_to(F.expand_dims(l2, 0), (batch_size, batch_size))
        ll = F.equal(l1_x, l2_x)
        labels = ll * (1 - self.smoothing_parameter) + (1 - ll) * self.smoothing_parameter / (batch_size - 1)
        ## now normalize rows to sum to 1.0
        labels = labels / F.broadcast_to(F.sum(labels, axis=1, keepdims=True), (batch_size, batch_size))
        if self.x2_downweight_idx >= 0:
            down_wt = len(mx.np.where(l2.as_np_ndarray != self.x2_downweight_idx)[0]) / batch_size
        else:
            down_wt = 1.0
        return labels, down_wt


    def _loss(self, F, x1, l1, x2, l2):
        """
        the function computes the kl divergence between the negative distances
        and the smoothed label matrix.
        """
        batch_size = x1.shape[0]
        labels, wt = self._compute_labels(F, l1, l2)
        distances = self._compute_distances(x1, x2)
        log_probabilities = F.log_softmax(-distances, axis=1)
        # multiply by the batch size to obtain the correct loss (gluon kl_loss averages instead of sum)
        return self.kl_loss(log_probabilities, labels.as_in_context(distances.context)) * batch_size * wt


    def hybrid_forward(self, F, x1, l1, x2, l2):
        return self._loss(F, x1, l1, x2, l2)    


