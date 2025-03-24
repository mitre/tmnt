# coding: utf-8
# Copyright (c) 2019-2021. The MITRE Corporation.
"""
Core Neural Net architectures for topic modeling.
"""

import math
import os
import numpy as np
import logging

from tmnt.distribution import LogisticGaussianDistribution
from tmnt.distribution import BaseDistribution
from torch import nn
from torch.nn.modules.loss import _Loss
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from typing import List, Tuple, Dict, Optional, Union, NoReturn

class BaseVAE(nn.Module):

    def __init__(self, vocab_size=2000, latent_distribution=LogisticGaussianDistribution(100, 20),
                 device='cpu', **kwargs):
        super(BaseVAE, self).__init__(**kwargs)        
        self.vocab_size = vocab_size        
        self.n_latent   = latent_distribution.n_latent
        self.enc_size   = latent_distribution.enc_size
        self.device = device
        self.embedding = None
        self.latent_distribution = latent_distribution
        self.decoder = nn.Linear(self.n_latent, self.vocab_size).to(device)
        self.npmi_with_diversity_loss : Optional[NPMILossWithDiversity] = None

    def initialize_bias_terms(self, wd_freqs: Optional[np.ndarray]):
        if wd_freqs is not None:
            freq_nd = wd_freqs + 1 # simple smoothing
            log_freq = np.log(freq_nd) - np.log(freq_nd.sum())
            with torch.no_grad():
                self.decoder.bias = nn.Parameter(torch.tensor(log_freq, dtype=torch.float32, device=self.device))
                self.decoder.bias.requires_grad_(False)

    def initialize_npmi_loss(self, npmi_mat, npmi_lambda=0.7, npmi_scale=100.0):
        t_npmi_mat = torch.Tensor(npmi_mat).to(self.device)
        self.npmi_with_diversity_loss = NPMILossWithDiversity(t_npmi_mat, device=self.device, npmi_lambda=npmi_lambda, npmi_scale=npmi_scale)

    def freeze_pre_encoder(self):
        pass

    def get_ordered_terms(self):
        """
        Returns the top K terms for each topic based on sensitivity analysis. Terms whose 
        probability increases the most for a unit increase in a given topic score/probability
        are those most associated with the topic.
        """
        z = torch.ones((self.n_latent,), device=self.device)
        jacobian = torch.autograd.functional.jacobian(self.decoder, z)
        sorted_j = jacobian.argsort(dim=0, descending=True)
        return sorted_j.cpu().numpy()
    
    def get_topic_vectors(self):
        """
        Returns unnormalized topic vectors
        """
        z = torch.ones((self.n_latent,), device=self.device)
        jacobian = torch.autograd.functional.jacobian(self.decoder, z)
        return jacobian.cpu().numpy()        


    def add_npmi_and_diversity_loss(self, cur_loss):
        if self.npmi_with_diversity_loss:
            z = torch.ones((self.n_latent,), device=self.device)
            jacobian = torch.autograd.functional.jacobian(self.decoder, z) 
            npmi_loss = self.npmi_with_diversity_loss(jacobian)
            npmi_loss = npmi_loss.sum()
            return (cur_loss + npmi_loss)
        else:
            return cur_loss


    def get_loss_terms(self, data, y, KL):
        rr = data * torch.log(y+1e-12)
        recon_loss = -(rr.sum(dim=1))
        i_loss = KL + recon_loss
        ii_loss = self.add_npmi_and_diversity_loss(i_loss)
        return ii_loss, recon_loss


class BowVAEModel(BaseVAE):
    """
    Defines the neural architecture for a bag-of-words topic model.

    Parameters:
        enc_dim (int): Number of dimension of input encoder (first FC layer)
        embedding_size (int): Number of dimensions for embedding layer
        n_encoding_layers (int): Number of layers used for the encoder. (default = 1)
        enc_dr (float): Dropout after each encoder layer. (default = 0.1)
        device (str): context device 
    """
    def __init__(self,
                 enc_dim, embedding_size, n_encoding_layers, enc_dr, 
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
        self.encoding_dims = [self.embedding_size] + [enc_dim for _ in range(n_encoding_layers)]
        self.embedding = torch.nn.Sequential()
        self.embedding.add_module("linear", torch.nn.Linear(self.vocab_size, self.embedding_size))
        self.embedding.add_module("tanh", torch.nn.Tanh())
        self.encoder   = self._get_encoder(self.encoding_dims, dr=enc_dr)
        if self.has_classifier:
            self.lab_dr = torch.nn.Dropout(self.classifier_dropout)
            self.classifier = torch.nn.Linear(self.n_latent, self.n_labels, bias=True)
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight.data)
            #torch.nn.init.xavier_uniform_(module.weight.data)
            
            
    def _get_encoder(self, dims, dr=0.1):
        encoder = torch.nn.Sequential()
        for i in range(len(dims)-1):
            encoder.add_module("linear_"+str(i), torch.nn.Linear(dims[i], dims[i+1]))
            encoder.add_module("soft_"+str(i), torch.nn.Softplus())
            if dr > 0.0:
                encoder.add_module("drop_"+str(i), torch.nn.Dropout(dr))
        return encoder

    def get_ordered_terms_encoder(self, dataloader, sample_size=-1):
        jacobians = np.zeros(shape=(self.n_latent, self.vocab_size))
        samples = 0
        
        def partial_network(data):
            emb_out = self.embedding(data)
            enc_out = self.latent_distribution.get_mu_encoding(self.encoder(emb_out), include_bn=True)
            return enc_out
        
        for bi, (data, _) in enumerate(dataloader):
            if sample_size > 0 and samples >= sample_size:
                print("Sample processed, exiting..")
                break
            samples += data.shape[0]
            x_data = x_data.to(device = self.device)
            x_data = torch.minimum(x_data, torch.tensor([1.0], device=self.device))
            jacobian = torch.autograd.functional.jacobian(partial_network, x_data)
            ss = jacobian.sum(dim=0).numpy()
            jacobian[bi] += ss
        sorted_j = (- jacobians).argsort(dim=1).transpose()
        return sorted_j

    def get_ordered_terms_per_item(self, dataloader, sample_size=-1):
        jacobian_list = [[] for i in range(self.n_latent)]
        samples = 0
        
        def partial_network(data):
            emb_out = self.embedding(data)
            enc_out = self.latent_distribution.get_mu_encoding(self.encoder(emb_out), include_bn=True)
            return enc_out
        
        for bi, (data, _) in enumerate(dataloader):
            if sample_size > 0 and samples >= sample_size:
                print("Sample processed, exiting..")
                break
            samples += data.shape[0]
            x_data = x_data.to(device = self.device)
            x_data = torch.minimum(x_data, torch.Tensor([1.0], device=self.device))
            jacobian = torch.autograd.functional.jacobian(partial_network, x_data)
            ss = jacobian.numpy()
            jacobian_list[bi] += list(ss)
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
    

    def run_encode(self, in_data, batch_size):
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
    

    def forward(self, data):
        data = data.to_dense()
        batch_size = data.shape[0]
        emb_out = self.embedding(data)
        #z, KL = self.run_encode(F, emb_out, batch_size)
        enc_out = self.encoder(emb_out)
        z, KL   = self.latent_distribution(enc_out, batch_size)
        xhat = self.decoder(z)
        y = torch.nn.functional.softmax(xhat, dim=1)
        ii_loss, recon_loss = \
            self.get_loss_terms(data, y, KL)
        if self.has_classifier:
            mu_out  = self.latent_distribution.get_mu_encoding(enc_out)
            classifier_outputs = self.classifier(self.lab_dr(mu_out))
        else:
            classifier_outputs = None
        return ii_loss, KL, recon_loss, classifier_outputs


class MetricBowVAEModel(BowVAEModel):

    def __init__(self, *args, **kwargs):
        self.kld_wt = 1.0
        super(MetricBowVAEModel, self).__init__(*args, **kwargs)


    def _get_elbo(self, bow, enc):
        batch_size = bow.shape[0]
        z, KL = self.latent_distribution(enc, batch_size)
        KL_loss = (KL * self.kld_wt)
        y = torch.nn.functional.softmax(self.decoder(z), dim=1)
        rec_loss = -torch.sum( bow * torch.log(y+1e-12), dim=1 )
        elbo = rec_loss + KL_loss
        return elbo, rec_loss, KL_loss

    def _get_encoding(self, data):
        return self.encoder( self.embedding(data) )

    def unpaired_input_forward(self, data):
        enc = self._get_encoding(data)
        elbo, rec_loss, kl_loss = self._get_elbo(data, enc)
        redundancy_loss = self.get_redundancy_penalty()
        return elbo, rec_loss, kl_loss, redundancy_loss

    def forward(self, F, data1, data2):
        enc1 = self._get_encoding(data1)
        enc2 = self._get_encoding(data2)
        mu1  = self.latent_distribution.get_mu_encoding(enc1)
        mu2  = self.latent_distribution.get_mu_encoding(enc2)
        elbo1, rec_loss1, KL_loss1 = self._get_elbo(data1, enc1)
        elbo2, rec_loss2, KL_loss2 = self._get_elbo(data2, enc2)        
        redundancy_loss = self.get_redundancy_penalty()
        return (elbo1 + elbo2), (rec_loss1 + rec_loss2), (KL_loss1 + KL_loss2), redundancy_loss, mu1, mu2


class NPMILossWithDiversity(nn.Module):

    def __init__(self, npmi_matrix: torch.Tensor, device: torch.device, k=20, npmi_lambda=0.7, npmi_scale=100.0, use_diversity_loss=True):
        super(NPMILossWithDiversity, self).__init__()
        self.npmi_lambda = npmi_lambda
        self.npmi_matrix = npmi_matrix
        self.npmi_scale  = npmi_scale
        self.use_diversity_loss = use_diversity_loss
        self.device = device
        self.k = k
        
    def _row_wise_normalize_inplace(self, x, mask=None):
        for row_idx, row in enumerate(x):
            if mask != None:
                row_mask = mask[row_idx]
                row = row[row_mask]
                x[row_idx][row_mask] = (row - row.min()) / (row.max() - row.min())
            else:
                row_min = row.min().item()
                row_max = row.max().item()
                x[row_idx] = (row - row_min)/(row_max - row_min)
        return x

    def _get_npmi_loss(self, jacobian):
        beta = jacobian.t()
        n_topics = beta.shape[0]
        self.npmi_matrix.fill_diagonal_(1)
        topk_idx = torch.topk(beta, self.k, dim=1)[1]
        topk_mask = torch.zeros_like(beta)
        for row_idx, indices in enumerate(topk_idx):
            topk_mask[row_idx, indices] = 1
        beta_mask = (1 - topk_mask) * -99999
        topk_mask = topk_mask.bool()
        topk_softmax_beta = torch.softmax(beta + beta_mask, dim=1)
        softmax_beta = torch.softmax(beta, dim=1)
        weighted_npmi = 1 - self._row_wise_normalize_inplace(torch.matmul(topk_softmax_beta.detach(), self.npmi_matrix))
        #print("Weighted_npmi sum = {}".format(weighted_npmi.sum()))
        npmi_loss = self.npmi_scale * (softmax_beta ** 2) * weighted_npmi
        if self.use_diversity_loss:
            diversity_mask = torch.zeros_like(beta).bool()
            for topic_idx in range(n_topics):
                other_rows_mask = torch.ones(n_topics).bool().to(self.device)
                other_rows_mask[topic_idx] = False
                diversity_mask[topic_idx] = topk_mask[other_rows_mask].sum(0) > 0
            npmi_loss = ( self.npmi_lambda * torch.masked_select(npmi_loss, diversity_mask)).sum() + \
                        ((1 - self.npmi_lambda) * torch.masked_select(npmi_loss, ~diversity_mask)).sum()
            npmi_loss *= 2
        return npmi_loss

    def forward(self, jacobian):
        return self._get_npmi_loss(jacobian)


class CoherenceRegularizer(nn.Module):

    ## Follows paper to add coherence loss: http://aclweb.org/anthology/D18-1096

    def __init__(self, coherence_pen=1.0, redundancy_pen=1.0):
        super(CoherenceRegularizer, self).__init__()
        self.coherence_pen = coherence_pen
        self.redundancy_pen = redundancy_pen
        

    def forward(self, w, emb):
        ## emb should have shape (D x V)
        ## w should have shape (V x K)
        # w NORM over columns
        w_min,_ = w.min(keepdim=True, dim=0)
        ww = w - w_min # ensure weights are non-negative
        w_norm_val = torch.norm(ww, keepdim=True, dim=0)
        emb_norm_val = torch.norm(emb, keepdim=True, dim=1)
        
        w_norm = ww / w_norm_val
        emb_norm = emb / emb_norm_val

        T = torch.matmul(emb_norm, w_norm)
        T_norm_vals = torch.norm(T, keepdim=True, dim=0)
        T_norm = T / T_norm_vals # (D x K)

        S = torch.matmul(emb_norm.t(), T_norm) # (V x K)
        C = -(S * w_norm).sum()
        ## diversity component
        D1 = torch.matmul(T_norm.t(), T_norm)
        D = D1.sum()
        return C * self.coherence_pen , D * self.redundancy_pen


class BaseSeqBowVED(BaseVAE):
    def __init__(self,
                 llm,
                 latent_dist: BaseDistribution,
                 num_classes=0,
                 dropout=0.0,
                 vocab_size=2000,
                 kld=0.1,
                 device='cpu',
                 use_pooling=True,
                 entropy_loss_coef=1000.0,
                 redundancy_reg_penalty=0.0, pre_trained_embedding = None):
        super(BaseSeqBowVED, self).__init__(device=device, vocab_size=vocab_size)
        self.n_latent = latent_dist.n_latent
        self.llm = llm
        self.kld_wt = kld
        self.has_classifier = num_classes >= 2
        self.num_classes = num_classes
        self.dropout = dropout
        self.redundancy_reg_penalty = redundancy_reg_penalty
        self.latent_distribution = latent_dist
        self.embedding = None
        self.decoder = nn.Linear(self.n_latent, vocab_size, bias=True).to(device)
        self.coherence_regularization = CoherenceRegularizer(0.0, self.redundancy_reg_penalty)
        self.use_pooling = use_pooling
        self.entropy_loss_coef = entropy_loss_coef
        if pre_trained_embedding is not None:
            self.embedding = nn.Linear(len(pre_trained_embedding.idx_to_vec),
                                           pre_trained_embedding.idx_to_vec[0].size, bias=False)
        #self.apply(self._init_weights)
        self.latent_distribution.apply(self._init_weights)
        self.decoder.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def _get_embedding(self, model_output, attention_mask):
        if self.use_pooling:
            token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            return model_output.last_hidden_state[:,0,:]
        
    def freeze_pre_encoder(self):
        for p in self.llm.parameters():
            p.requires_grad = False
        self.latent_distribution.freeze_pre_encoder()

    def get_ordered_terms(self):
        """
        Returns the top K terms for each topic based on sensitivity analysis. Terms whose 
        probability increases the most for a unit increase in a given topic score/probability
        are those most associated with the topic.
        """
        z = torch.ones((self.n_latent,), device=self.device)
        jacobian = torch.autograd.functional.jacobian(self.decoder, z)
        sorted_j = jacobian.argsort(dim=0, descending=True)
        return sorted_j.cpu().numpy()
            

    def get_redundancy_penalty(self):
        w = self.decoder.weight.data
        emb = self.embedding.weight.data if self.embedding is not None else w.transpose(0,1)
        _, redundancy_loss = self.coherence_regularization(w, emb)
        return redundancy_loss
    
    def _get_latent_sparsity_term(self, encoding):
        as_distribution = Categorical(probs=encoding) if self.latent_distribution.on_simplex else Categorical(logits=encoding)
        return as_distribution.entropy()
    
    def _get_elbo(self, bow, enc):
        z, KL = self.latent_distribution(enc, bow.size()[0])
        KL_loss = (KL * self.kld_wt)
        dec = self.decoder(z)
        y = torch.nn.functional.softmax(dec, dim=1)
        rec_loss = -torch.sum( bow.to_dense() * torch.log(y+1e-12), dim=1 )
        entropy_loss = self._get_latent_sparsity_term(z)
        elbo = rec_loss + KL_loss + (entropy_loss * self.entropy_loss_coef)
        return elbo, rec_loss, KL_loss, entropy_loss

    def forward_encode(self, input_ids, attention_mask):
        llm_output = self.llm(input_ids, attention_mask)    
        cls_vec = self._get_embedding(llm_output, attention_mask)
        return self.latent_distribution.get_mu_encoding(cls_vec)
    

class SeqBowVED(BaseSeqBowVED):
    def __init__(self, *args, **kwargs):
        super(SeqBowVED, self).__init__(*args, **kwargs)
        if self.has_classifier:
            self.classifier = torch.nn.Sequential()
            self.classifier.add_module("dr", nn.Dropout(self.dropout).to(self.device))
            self.classifier.add_module("l_out", nn.Linear(self.n_latent, self.num_classes).to(self.device))

        
    def forward(self, input_ids, attention_mask, bow=None):  # pylint: disable=arguments-differ
        llm_output = self.llm(input_ids, attention_mask)
        cls_vec = self._get_embedding(llm_output, attention_mask)
        return self.forward_with_cached_encoding(cls_vec, bow)

    def forward_with_cached_encoding(self, enc, bow):
        elbo, rec_loss, KL_loss = 0.0, 0.0, 0.0
        if bow is not None:
            elbo, rec_loss, KL_loss, entropy_loss = self._get_elbo(bow, enc)
        if self.has_classifier:
            z_mu = self.latent_distribution.get_mu_encoding(enc)            
            classifier_outputs = self.classifier(z_mu)
        else:
            classifier_outputs = None
        ii_loss = self.add_npmi_and_diversity_loss(elbo)
        redundancy_loss = ii_loss  #self.get_redundancy_penalty()
        return ii_loss, rec_loss, KL_loss, redundancy_loss, classifier_outputs


class MetricSeqBowVED(BaseSeqBowVED):
    def __init__(self, *args, **kwargs):
        super(MetricSeqBowVED, self).__init__(*args, **kwargs)

    def unpaired_input_forward(self, in1, mask1, bow1):
        llm_output = self.llm(in1, mask1)
        cls_vec = self._get_embedding(llm_output, mask1)
        elbo1, rec_loss1, KL_loss1, entropy_loss = self._get_elbo(bow1, cls_vec)
        redundancy_loss = entropy_loss # self.get_redundancy_penalty()
        return elbo1, rec_loss1, KL_loss1, redundancy_loss

    def forward(self, in1, mask1, bow1, in2, mask2, bow2):
        llm_out1 = self.llm(in1, mask1)
        llm_out2 = self.llm(in2, mask2)
        enc1 = self._get_embedding(llm_out1, mask1)
        enc2 = self._get_embedding(llm_out2, mask2)
        elbo1, rec_loss1, KL_loss1, entropy_loss1 = self._get_elbo(bow1, enc1)
        elbo2, rec_loss2, KL_loss2, entropy_loss2 = self._get_elbo(bow2, enc2)
        elbo = elbo1 + elbo2
        rec_loss = rec_loss1 + rec_loss2
        KL_loss = KL_loss1 + KL_loss2
        #z_mu1 = self.latent_distribution.get_mu_encoding(enc2)
        #z_mu2 = self.latent_distribution.get_mu_encoding(enc2)
        redundancy_loss = entropy_loss1 + entropy_loss2 #self.get_redundancy_penalty()
        #return elbo, rec_loss, KL_loss, redundancy_loss, z_mu1, z_mu2
        return elbo, rec_loss, KL_loss, redundancy_loss, enc1, enc2


class GeneralizedSDMLLoss(_Loss):
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
        self.kl_loss = nn.KLDivLoss(size_average=False, reduction='batchmean')
        self.smoothing_parameter = smoothing_parameter # Smoothing probability mass
        self.x2_downweight_idx = x2_downweight_idx

    def _compute_distances(self, x1, x2):
        """
        This function computes the euclidean distance between every vector
        in the two batches in input.
        """

        # extracting sizes expecting [batch_size, dim]
        assert x1.size() == x2.size()
        batch_size, dim = x1.size()
        # expanding both tensor form [batch_size, dim] to [batch_size, batch_size, dim]
        x1_ = x1.unsqueeze(1).broadcast_to([batch_size, batch_size, dim])
        x2_ = x2.unsqueeze(0).broadcast_to([batch_size, batch_size, dim])
        # pointwise squared differences
        squared_diffs = (x1_ - x2_)**2
        # sum of squared differences distance
        return squared_diffs.sum(axis=2)


    def _compute_labels(self, l1: torch.Tensor, l2: torch.Tensor):
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
        batch_size = l1.size()[0]
        l1_x = l1.unsqueeze(1).expand(batch_size, batch_size)
        l2_x = l2.unsqueeze(0).expand(batch_size, batch_size)
        #l1_x = F.broadcast_to(F.expand_dims(l1, 1), (batch_size, batch_size))
        #l2_x = F.broadcast_to(F.expand_dims(l2, 0), (batch_size, batch_size))
        ll = torch.eq(l1_x, l2_x)
        labels = ll * (1 - self.smoothing_parameter) + (~ll) * self.smoothing_parameter / (batch_size - 1)
        ## now normalize rows to sum to 1.0
        labels = labels / labels.sum(axis=1,keepdim=True).expand(batch_size, batch_size)
        if self.x2_downweight_idx >= 0:
            #down_wt = len(mx.np.where(l2.as_np_ndarray != self.x2_downweight_idx)[0]) / batch_size
            down_wt = len(np.where(l2 != self.x2_downweight_idx)[0]) / batch_size
        else:
            down_wt = 1.0
        return labels, down_wt


    def _loss(self, x1: torch.Tensor, l1: torch.Tensor, x2: torch.Tensor, l2: torch.Tensor):
        """
        the function computes the kl divergence between the negative distances
        and the smoothed label matrix.
        """
        labels, wt = self._compute_labels(l1, l2)
        distances = self._compute_distances(x1, x2)
        log_probabilities = torch.log_softmax(-distances, dim=1)
        # multiply by the batch size to obtain the sum loss (kl_loss averages instead of sum)
        kl = self.kl_loss(log_probabilities, labels.to(distances.device)) * wt
        return kl 


    def forward(self, x1, l1, x2, l2):
        return self._loss(x1, l1, x2, l2)    



class MultiNegativeCrossEntropyLoss(_Loss):
    """
    Inputs:
        - **x1**: Minibatch of data points with shape (batch_size, vector_dim)
        - **x2**: Minibatch of data points with shape (batch_size, vector_dim)
          Each item in x1 is a positive sample for the items with the same label in x2
          That is, x1[0] and x2[0] form a positive pair iff label(x1[0]) = label(x2[0])
          All data points in different rows should be decorrelated

    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, smoothing_parameter=0.1, metric_loss_temp=0.1, batch_axis=0, **kwargs):
        super(MultiNegativeCrossEntropyLoss, self).__init__(batch_axis, **kwargs)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.smoothing_parameter = smoothing_parameter # Smoothing probability mass
        self.metric_loss_temp = metric_loss_temp

    def _compute_distances(self, x1, x2):
        """
        This function computes the cosine distance between every vector
        in the two batches in input.
        """

        # extracting sizes expecting [batch_size, dim]
        assert x1.size() == x2.size()
        # expanding both tensor form [batch_size, dim] to [batch_size, batch_size, dim]
        x1_norm = torch.nn.functional.normalize(x1, p=2, dim=1)
        x2_norm = torch.nn.functional.normalize(x2, p=2, dim=1)
        return torch.mm(x1_norm, x2_norm.transpose(0, 1)) 


    def _compute_labels(self, l1: torch.Tensor, l2: torch.Tensor):
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
        batch_size = l1.size()[0]
        l1_x = l1.unsqueeze(1).expand(batch_size, batch_size)
        l2_x = l2.unsqueeze(0).expand(batch_size, batch_size)
        #l1_x = F.broadcast_to(F.expand_dims(l1, 1), (batch_size, batch_size))
        #l2_x = F.broadcast_to(F.expand_dims(l2, 0), (batch_size, batch_size))
        ll = torch.eq(l1_x, l2_x)
        labels = ll * (1 - self.smoothing_parameter) + (~ll) * self.smoothing_parameter / (batch_size - 1)
        ## now normalize rows to sum to 1.0
        labels = labels / labels.sum(axis=1,keepdim=True).expand(batch_size, batch_size)
        return labels


    def _loss(self, x1: torch.Tensor, l1: torch.Tensor, x2: torch.Tensor, l2: torch.Tensor):
        """
        the function computes the kl divergence between the negative distances
        and the smoothed label matrix.
        """
        labels = self._compute_labels(l1, l2)
        distances = self._compute_distances(x1, x2) / self.metric_loss_temp
        # multiply by the batch size to obtain the sum loss (kl_loss averages instead of sum)
        return self.cross_entropy_loss(distances, labels.to(distances.device))


    def forward(self, x1, l1, x2, l2):
        return self._loss(x1, l1, x2, l2)    


class CrossBatchCosineSimilarityLoss(_Loss):
    """
    Inputs:
        - **x1**: Minibatch of data points with shape (batch_size, vector_dim)
        - **x2**: Minibatch of data points with shape (batch_size, vector_dim)
          Each item in x1 is a positive sample for the items with the same label in x2

    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, teacher_mode='rand', batch_axis=0, **kwargs):
        super(CrossBatchCosineSimilarityLoss, self).__init__(batch_axis, **kwargs)
        self.loss_fn = nn.MSELoss() 
        self.teacher_mode = teacher_mode
        
    def cosine_sim(self, a: Tensor, b: Tensor) -> Tensor:
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def _loss(self, x1: torch.Tensor, l1: torch.Tensor, x2: torch.Tensor, l2: torch.Tensor):
        scores = self.cosine_sim(x1,x2) 
        if self.teacher_mode == 'right':
            labels = self.cosine_sim(x2,x2).detach()
        elif self.teacher_mode == 'left':
            labels = self.cosine_sim(x1,x1).detach()
        else:
            if np.random.randint(2):
                labels = self.cosine_sim(x2,x2).detach() 
            else:
                labels = self.cosine_sim(x1,x1).detach()
        return self.loss_fn(scores, labels)

    def forward(self, x1, l1, x2, l2):
        return self._loss(x1, l1, x2, l2)    
