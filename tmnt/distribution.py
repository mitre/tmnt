#coding: utf-8
# Copyright (c) 2019-2021 The MITRE Corporation.
"""
Variational latent distributions (e.g. Gaussian, Logistic Gaussian)
"""

import math
import numpy as np
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions import VonMises
from torch.nn import Sequential
import torch
from scipy import special as sp
import torch


__all__ = ['BaseDistribution', 'GaussianDistribution', 'GaussianUnitVarDistribution', 'LogisticGaussianDistribution',
           'VonMisesDistribution']


class BaseDistribution(nn.Module):
    
    def __init__(self, enc_size, n_latent, device, on_simplex=False):
        super(BaseDistribution, self).__init__()
        self.n_latent = n_latent
        self.enc_size = enc_size
        self.device = device
        self.mu_encoder = nn.Linear(enc_size, n_latent).to(device)
        #self.mu_encoder = Sequential(self.mu_proj, nn.Softplus().to(device))
        self.mu_bn = nn.BatchNorm1d(n_latent, momentum = 0.8, eps=0.0001).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.softplus = nn.Softplus().to(device)      
        self.on_simplex = on_simplex
        #self.mu_bn.collect_params().setattr('grad_req', 'null')

    ## this is required by most priors
    def _get_gaussian_sample(self, mu, lv, batch_size):
        eps = Normal(torch.zeros(batch_size, self.n_latent), 
                     torch.ones(batch_size, self.n_latent)).sample().to(self.device)
        return (mu + torch.exp(0.5*lv).to(self.device) * eps)

    ## this is required by most priors
    def _get_unit_var_gaussian_sample(self, mu, batch_size):
        eps = Normal(torch.zeros(batch_size, self.n_latent), torch.ones(batch_size, self.n_latent)).sample()
        return (mu + eps).to(self.device)
    
    def get_mu_encoding(self, data, include_bn):
        raise NotImplemented 




class GaussianDistribution(BaseDistribution):
    """Gaussian latent distribution with diagnol co-variance.

    Parameters:
        n_latent (int): Dimentionality of the latent distribution
        device (device): Torch computational context (cpu or gpu[id])
        dr (float): Dropout value for dropout applied post sample. optional (default = 0.2)
    """
    def __init__(self, enc_size, n_latent, device='cpu', dr=0.2):
        super(GaussianDistribution, self).__init__(enc_size, n_latent, device)
        self.lv_encoder = nn.Linear(enc_size, n_latent).to(device) 
        self.lv_bn = nn.BatchNorm1d(n_latent, momentum = 0.8, eps=0.001).to(device)
        self.post_sample_dr_o = nn.Dropout(p=dr)        

    def _get_kl_term(self, mu, lv):
        ww = self.lv_encoder.weight.data
        return (-0.5 * torch.sum(1 + lv - mu*mu - torch.exp(lv), 1)).to(self.device)

    def forward(self, data, batch_size):
        """Generate a sample according to the Gaussian given the encoder outputs
        """
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)
        mu_bn = self.softplus(mu_bn)
        lv = self.lv_encoder(data)
        lv_bn = self.lv_bn(lv)
        z = self._get_gaussian_sample(mu_bn, lv_bn, batch_size)
        KL = self._get_kl_term(mu_bn, lv_bn)
        z = self.post_sample_dr_o(z)
        return z, KL
    
    def get_mu_encoding(self, data, include_bn=True, normalize=False):
        """Provide the distribution mean as the natural result of running the full encoder
        
        Parameters:
            data (:class:`mxnet.ndarray.NDArray`): Output of pre-latent encoding layers
        Returns:
            encoding (:class:`mxnet.ndarray.NDArray`): Encoding vector representing unnormalized topic proportions
        """
        enc = self.mu_encoder(data)
        if include_bn:
            enc = self.mu_bn(enc)
        mu = self.softplus(enc) if normalize else enc
        return mu
        


class GaussianUnitVarDistribution(BaseDistribution):
    """Gaussian latent distribution with fixed unit variance.

    Parameters:
        n_latent (int): Dimentionality of the latent distribution
        device (device): Torch computational context (cpu or gpu[id])
        dr (float): Dropout value for dropout applied post sample. optional (default = 0.2)
    """
    def __init__(self, n_latent, device='cpu', dr=0.2, var=1.0):
        super(GaussianUnitVarDistribution, self).__init__(n_latent,device)
        self.variance = torch.tensor([var], device=device)
        self.log_variance = torch.log(self.variance)
        with self.name_scope():
            self.post_sample_dr_o = torch.nn.Dropout(dr)

    def _get_kl_term(self, mu):
        return (-0.5 * torch.sum(1.0 + self.log_variance - mu*mu - self.variance, axis=1)).to(self.device)

    def forward(self, data, batch_size):
        """Generate a sample according to the unit variance Gaussian given the encoder outputs
        """
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)
        mu_bn = self.softplus(mu_bn)
        z = self._get_gaussian_sample(mu_bn, self.log_variance, batch_size)
        KL = self._get_kl_term(mu_bn)
        return self.post_sample_dr_o(z), KL
    
    def get_mu_encoding(self, data, include_bn=True, normalize=False):
        """Provide the distribution mean as the natural result of running the full encoder
        
        Parameters:
            data (:class:`mxnet.ndarray.NDArray`): Output of pre-latent encoding layers
        Returns:
            encoding (:class:`mxnet.ndarray.NDArray`): Encoding vector representing unnormalized topic proportions
        """
        enc = self.mu_encoder(data)
        if include_bn:
            enc = self.mu_bn(enc)
        mu = self.softplus(enc) if normalize else enc 
        return mu
        

class LogisticGaussianDistribution(BaseDistribution):
    """Logistic normal/Gaussian latent distribution with specified prior

    Parameters:
        n_latent (int): Dimentionality of the latent distribution
        device (device): Torch computational context (cpu or gpu[id])
        dr (float): Dropout value for dropout applied post sample. optional (default = 0.2)
        alpha (float): Value the determines prior variance as 1/alpha - (2/n_latent) + 1/(n_latent^2)
    """
    def __init__(self, enc_size, n_latent, device='cpu', dr=0.1, alpha=1.0):
        super(LogisticGaussianDistribution, self).__init__(enc_size, n_latent, device, on_simplex=True)
        self.alpha = alpha

        prior_var = 1 / self.alpha - (2.0 / n_latent) + 1 / (self.n_latent * self.n_latent)
        self.prior_var = torch.tensor([prior_var], device=device)
        self.prior_logvar = torch.tensor([math.log(prior_var)], device=device)

        self.lv_encoder = nn.Linear(enc_size, n_latent).to(device)
        self.lv_bn = nn.BatchNorm1d(n_latent, momentum = 0.8, eps=0.001).to(device)
        self.post_sample_dr_o = nn.Dropout(dr)
        #self.lv_bn.collect_params().setattr('grad_req', 'null')        

    def _get_kl_term(self, mu, lv):
        posterior_var = torch.exp(lv)
        delta = mu
        dt = torch.div(delta * delta, self.prior_var)
        v_div = torch.div(posterior_var, self.prior_var)
        lv_div = self.prior_logvar - lv
        return (0.5 * (torch.sum((v_div + dt + lv_div), 1) - self.n_latent)).to(self.device)

    def forward(self, data, batch_size):
        """Generate a sample according to the logistic Gaussian latent distribution given the encoder outputs
        """
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)        
        lv = self.lv_encoder(data)
        lv_bn = self.lv_bn(lv)
        z_p = self._get_gaussian_sample(mu_bn, lv_bn, batch_size)
        KL = self._get_kl_term(mu, lv)
        z = self.post_sample_dr_o(z_p)
        return self.softmax(z), KL

    def get_mu_encoding(self, data, include_bn=True, normalize=False):
        """Provide the distribution mean as the natural result of running the full encoder
        
        Parameters:
            data (:class:`mxnet.ndarray.NDArray`): Output of pre-latent encoding layers
        Returns:
            encoding (:class:`mxnet.ndarray.NDArray`): Encoding vector representing unnormalized topic proportions
        """
        enc = self.mu_encoder(data)
        if include_bn:
            enc = self.mu_bn(enc)
        mu = self.softmax(enc) if normalize else enc
        return mu
        
    
class VonMisesDistribution(BaseDistribution):
    
    def __init__(self, enc_size, n_latent, kappa=100.0, dr=0.1, device='cpu'):
        super(VonMisesDistribution, self).__init__(enc_size, n_latent, device, on_simplex=False)
        self.device = device
        self.kappa = kappa
        self.kld_v = torch.tensor(VonMisesDistribution._vmf_kld(self.kappa, self.n_latent), device=device)


    @staticmethod
    def _vmf_kld(k, d):
        return np.array([(k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k))
                          + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k))
                          - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2).real])


    def forward(self, data, batch_size):
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)
        mu_bn = self.softplus(mu_bn)
        z_p = VonMises(mu_bn, self.kappa).sample()
        kld = self.kld_v.expand(batch_size)
        return z_p, kld

    def get_mu_encoding(self, data, include_bn=True, normalize=False):
        """Provide the distribution mean as the natural result of running the full encoder
        
        Parameters:
            data (:class:`mxnet.ndarray.NDArray`): Output of pre-latent encoding layers
        Returns:
            encoding (:class:`mxnet.ndarray.NDArray`): Encoding vector representing unnormalized topic proportions
        """
        enc = self.mu_encoder(data)
        if include_bn:
            enc = self.mu_bn(enc)
        mu = self.softplus(enc) if normalize else enc
        return mu
        
    

class Projection(BaseDistribution):

    def __init__(self, enc_size, n_latent, device='cpu'):
        super(Projection, self).__init__(enc_size, n_latent, device)
        

    def forward(self, data, batch_size):
        mu = self.mu_encoder(data)
        mu_bn = self.mu_bn(mu)
        kld = torch.zeros(batch_size).to(self.device)
        return mu_bn, kld

    def get_mu_encoding(self, data, include_bn=True, normalize=False):
        """Provide the distribution mean as the natural result of running the full encoder
        
        Parameters:
            data (:class:`mxnet.ndarray.NDArray`): Output of pre-latent encoding layers
        Returns:
            encoding (:class:`mxnet.ndarray.NDArray`): Encoding vector representing unnormalized topic proportions
        """
        enc = self.mu_encoder(data)
        if include_bn:
            enc = self.mu_bn(enc)
        return enc
        
        
    

    
