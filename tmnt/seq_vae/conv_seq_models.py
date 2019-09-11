# coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""


import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock, Block
from mxnet.gluon import nn
from tmnt.distributions import LogisticGaussianLatentDistribution, GaussianLatentDistribution, HyperSphericalLatentDistribution
import math


class SeqVAE(HybridBlock):
    
    def __init__(self, embed_dim=1024, vocab_dim=10000, model_ctx=mx.cpu(), n_latent=256, max_sent_len=32,
                 filter_size=4, num_filters=64, batch_size=8, decreasing=False, increasing=False, kld_wt=1.0,
                 prefix=None, params=None):
            
        super(SeqVAE, self).__init__(prefix=prefix, params=params)
        self.soft_zero = 1e-10
        self.embedding = nn.Embedding(input_dim=vocab_dim, output_dim=embed_dim, weight_initializer=mx.init.Normal(0.1))
        self.encoder = EncoderBlock(embed_dim=embed_dim, n_latent=n_latent, sent_size=max_sent_len,
                                    num_filters=num_filters, increasing=increasing, decreasing=decreasing)
        self.decoder = DecoderBlock(embed_dim=embed_dim, n_latent=n_latent, sent_size=max_sent_len,
                                    num_filters=num_filters, batch_size=batch_size,
                                    increasing=increasing, decreasing=decreasing)
        self.batch_size = batch_size
        self.n_latent = n_latent
        self.model_ctx=model_ctx
        self.max_sent_len = max_sent_len
        self.kld_wt = kld_wt
        self.vocab_dim = vocab_dim
        self.encoder_input_dim = embed_dim
        self.inv_embed = InverseEmbed(batch_size, max_sent_len, embed_dim, params=self.embedding.params)
        self.ce_loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=-1, from_logits=True)

    
    def __call__(self, in_data):
        return super(SeqVAE, self).__call__(in_data)


    def hybrid_forward(self, F, x_toks):
        x1 = self.embedding(x_toks)  ## x_toks ~ (batch, 1, ids)
        ####  -- may want to do some kind of max normalization here to prevent these from moving around too much
        #x1_norm = F.norm(x1)
        #x2 = x1 / x1_norm
        ##
        x = F.reshape(x1,(self.batch_size,1,self.max_sent_len,-1))
        enc_out_p = self.encoder(x)
        enc_out = F.reshape(enc_out_p, (self.batch_size, self.n_latent * 2))  ## *2 for variational model
        mu_lv = F.split(enc_out, axis=1, num_outputs=2) ## split in half along final dimension
        mu = mu_lv[0]
        lv = mu_lv[1]
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=self.model_ctx)
        z = mu + F.exp(0.5*lv)*eps
        y = self.decoder(z)
        KL = 0.5*F.sum(1+lv-mu*mu-F.exp(lv),axis=1)
        
        y_norm = F.norm(y, axis=-1, keepdims=True)   # so we can normalize by this norm
        rec_y_1 = F.broadcast_div(y, y_norm) ## y / y_norm
        rec_y = F.reshape(rec_y_1, (self.batch_size, self.max_sent_len, self.encoder_input_dim))  ## shape of last dim is |V|
        ## does a matrix mult: rec_y = (32, 64, 300), shape mm = (32, 300, 25002)
        prob_logits = self.inv_embed(rec_y)
        log_prob = F.log_softmax(prob_logits)
        loss = self.ce_loss_fn(log_prob, x_toks) - (KL * self.kld_wt)
        return loss, log_prob


class EncoderBlock(HybridBlock):
    """
    embed_dim : int - number of dimensions for each input position (e.g. 1024 for single Elmo layer)
    n_latent : int    - number of dimensions for latent Guassian/code 
    sent_size : int   - maximum sentence size; shorter sentences are assumed to be padded (longer truncated)
    filter_size : int - height of the convolution kernels for intermediate layers
    num_filters : int - number of filters for the first convolution layer
    decreasing : bool - if true, then the number of filters is halved for each subsequent layer in the encoder (instead of doubled)
    activation : str  - string denoting activation type for intermediate convolutional layers (default 'relu')    
    """
    def __init__(self, embed_dim=1024, n_latent=256, sent_size = 16, filter_size=4, num_filters=64, variational=True, decreasing=False,
                 increasing=False,activation='relu', prefix=None, params=None):
        super(EncoderBlock, self).__init__(prefix=prefix, params=params)

        num_layers = int(math.log(sent_size, 2) - 1)        
        filters = [int(num_filters / math.pow(2,i)) if decreasing else  \
                   int(num_filters * math.pow(2,i)) if increasing else num_filters for i in range(num_layers-1)]
        filters.append(n_latent * 2 if variational else n_latent) ## final filter size has to map to encoding size (x2 if variational)
        strides     = [1 if i == num_layers-1 else 2 for i in range(num_layers)]
        widths      = [embed_dim if i == 0 else 1 for i in range(num_layers)]
        padding     = [0 if i == num_layers-1 else 1 for i in range(num_layers)]
        activations = [None if i == num_layers-1 else 'relu' for i in range(num_layers)]
        in_channels = [1 if i == 0 else filters[i-1] for i in range(num_layers)]
        self.enc_layers = nn.HybridSequential()
        with self.name_scope():
            for i in range(num_layers):
                self.enc_layers.add(
                    nn.Conv2D(filters[i], (int(filter_size), widths[i]), strides=(strides[i], strides[i]), padding=(padding[i], 0),
                              in_channels=in_channels[i], activation=activations[i], use_bias=False))
                if i < num_layers-1: ## if not the last layer, add batch norm + ReLU
                    self.enc_layers.add(nn.Activation(activation=activations[i]))                    
                    self.enc_layers.add(nn.BatchNorm(axis=1, center=True, scale=True, in_channels=in_channels[i+1]))

    def __call__(self, x):
        return super(EncoderBlock, self).__call__(x)

    def hybrid_forward(self, F, x):
        return self.enc_layers(x)




class DecoderBlock(HybridBlock):
    """
    embed_dim : int - number of dimensions for each input position (e.g. 1024 for single Elmo layer)
    n_latent : int    - number of dimensions for latent Guassian/code 
    sent_size : int   - maximum sentence size; shorter sentences are assumed to be padded (longer truncated)
    filter_size : int - height of the convolution kernels for intermediate layers
    num_filters : int - number of filters for the first convolution layer
    decreasing : bool - if true, then the number of filters is halved for each subsequent layer in the decoder
    increasing : bool - if true, then the number of filters is doubled for each subsequent layer in the decoder 
    activation : str  - string denoting activation type for intermediate convolutional layers (default 'relu')    
    """
    def __init__(self, embed_dim=1024, n_latent=256, sent_size = 32, filter_size=4, num_filters=64, batch_size=8,
                 decreasing=False, increasing=False,
                 activation='relu'):
        super(DecoderBlock, self).__init__()
        self._batch_size = batch_size
        self._n_latent = n_latent
        self._sent_size = sent_size
        num_layers = int(math.log(sent_size,2) - 1)
        self.num_layers = num_layers
        ## NOTE: Decoder is operating inversely to encoder, so "decreasing" actually is increasing for the decoder
        filters = [int(num_filters * math.pow(2,i)) if decreasing else  \
                   int(num_filters / math.pow(2,i)) if increasing else num_filters for i in range(num_layers-1)]
        filters.append(1) ## final filter size is one
        strides = [1 if i == 0 else 2 for i in range(num_layers)]
        padding = [0 if i == 0 else 1 for i in range(num_layers)]
        heights = [int(filter_size) for i in range(num_layers)]
        widths = [1 if i < num_layers-1 else embed_dim for i in range(num_layers)]
        self.widths = widths
        self.heights = heights
        self.strides = strides
        activations = [None if i == num_layers - 1 else 'relu' for i in range(num_layers)]
        in_channels = [self._n_latent if i == 0 else filters[i-1] for i in range(num_layers)]
        self.dec_layers = nn.HybridSequential()
        with self.dec_layers.name_scope():
            for i in range(num_layers):
                self.dec_layers.add(
                    nn.Conv2DTranspose(filters[i], (heights[i], widths[i]), strides=(strides[i],1), padding=(padding[i],0),
                                       output_padding=(0,0),
                                               in_channels=in_channels[i], activation=None)) 
                if i < num_layers-1: ## if not the last layer, add batch norm
                    self.dec_layers.add(nn.BatchNorm(axis=1, center=True, scale=True, in_channels=in_channels[i+1]))                    
                self.dec_layers.add(nn.Activation(activation='relu'))

                    
    def __call__(self, x):
        return super(DecoderBlock, self).__call__(x)

    def hybrid_forward(self, F, x):
        x = F.reshape(x, (self._batch_size, self._n_latent, 1, 1)) ## back to rank 4 tensor for conv operations
        return self.dec_layers(x)



