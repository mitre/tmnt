# codeing: utf-8

__all__ = ['BertTransVAE']

import math
import os
import numpy as np

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.gluon.loss import L2Loss
from mxnet.gluon.block import HybridBlock, Block
from gluonnlp.model import TransformerEncoderCell, TransformerEncoder
from tmnt.seq_vae.seq_models import InverseEmbed
from tmnt.distributions import LogisticGaussianLatentDistribution, GaussianLatentDistribution, HyperSphericalLatentDistribution

class PureTransformerVAE(Block):

    def __init__(self, vocabulary, latent_distrib='vmf', num_units=512,
                 n_latent=256, max_sent_len=64, transformer_layers=6,
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_temp=0.01, ctx = mx.cpu(),
                 increasing=True, decreasing=False,
                 prefix=None, params=None):
        super(PureTransformerVAE, self).__init__(prefix=prefix, params=params)
        self.kld_wt = kld
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.max_sent_len = max_sent_len
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.wd_embed_dim = len(vocabulary.embedding.idx_to_vec[0]) # word embedding length
        self.vocab_size = len(vocabulary.embedding.idx_to_token)
        self.latent_distrib = latent_distrib
        self.num_units = num_units
        with self.name_scope():
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=kappa, ctx=self.model_ctx)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarLatentDistribution(n_latent, ctx)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
            self.embedding = nn.Embedding(self.vocab_size, self.wd_embed_dim)
            #self.projection = gluon.nn.Dense(
            self.encoder = TransformerEncoder(self.wd_embed_dim, self.num_units, n_layers=transformer_layers, n_latent=n_latent, sent_size = max_sent_len,
                                              batch_size = batch_size, ctx = ctx)
            self.decoder = TransformerDecoder(wd_embed_dim=self.wd_embed_dim, num_units=self.num_units,
                                              n_layers=transformer_layers, n_latent=n_latent, sent_size = max_sent_len,
                                              batch_size = batch_size, ctx = ctx)
            self.out_embedding = gluon.nn.Embedding(input_dim=self.vocab_size, output_dim=self.wd_embed_dim)
            self.inv_embed = InverseEmbed(batch_size, max_sent_len, self.wd_embed_dim, temp=wd_temp, ctx=self.model_ctx, params = self.out_embedding.params)
            self.ce_loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=-1, from_logits=True)
        self.embedding.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
        self.out_embedding.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)        
        self.inv_embed.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
        self.out_embedding.weight.set_data(self.vocabulary.embedding.idx_to_vec)
        self.embedding.weight.set_data(self.vocabulary.embedding.idx_to_vec)
        #self.embedding.collect_params().setattr('grad_req', 'null')  ## force embedding weights to stay fixed
        #self.inv_embed.collect_params().setattr('grad_req', 'null')  ## force embedding weights to stay fixed
        

    def __call__(self, wp_toks):
        return super(PureTransformerVAE, self).__call__(wp_toks)

    def set_kl_weight(self, epoch, max_epochs):
        burn_in = int(max_epochs / 10)
        eps = 1e-6
        if epoch > burn_in:
            self.kld_wt = ((epoch - burn_in) / (max_epochs - burn_in)) + eps
        else:
            self.kld_wt = eps
        return self.kld_wt


    def forward(self, toks):
        embedded = self.embedding(toks)
        enc = self.encoder(embedded)
        
        z, KL = self.latent_dist(enc, self.batch_size)
        y = self.decoder(z)
        
        #y_norm = mx.nd.norm(y, axis=-1, keepdims=True)   # so we can normalize by this norm
        #rec_y = mx.nd.broadcast_div(y, y_norm) ## y / y_norm
        #rec_y = mx.nd.reshape(rec_y_1, (self.batch_size, self.max_sent_len, self.wd_embed_dim))

        prob_logits = self.inv_embed(y)
        log_prob = mx.nd.log_softmax(prob_logits)
        ## reconstruction loss is weighted combo of cross entropy over vocab and cosine loss over embeddings
        recon_loss = self.ce_loss_fn(log_prob, toks)
        kl_loss = (KL * self.kld_wt)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss, log_prob
    

class BertTransVAE(Block):
    def __init__(self, bert_base, latent_distrib='vmf',
                 wd_embed_dim=300, num_units=512, n_latent=256, max_sent_len=64, transformer_layers=6,
                 kappa = 100.0,
                 batch_size=16, kld=0.1, wd_temp=0.01, ctx = mx.cpu(),
                 increasing=True, decreasing=False,
                 prefix=None, params=None):
        super(BertTransVAE, self).__init__(prefix=prefix, params=params)
        self.kld_wt = kld
        self.bert = bert_base
        self.n_latent = n_latent
        self.model_ctx = ctx
        self.max_sent_len = max_sent_len
        self.batch_size = batch_size
        self.wd_embed_dim = wd_embed_dim
        self.latent_distrib = latent_distrib
        with self.name_scope():
            if latent_distrib == 'logistic_gaussian':
                self.latent_dist = LogisticGaussianLatentDistribution(n_latent, ctx)
            elif latent_distrib == 'vmf':
                self.latent_dist = HyperSphericalLatentDistribution(n_latent, kappa=kappa, ctx=self.model_ctx)
            elif latent_distrib == 'gaussian':
                self.latent_dist = GaussianLatentDistribution(n_latent, ctx)
            elif latent_distrib == 'gaussian_unitvar':
                self.latent_dist = GaussianUnitVarLatentDistribution(n_latent, ctx)
            else:
                raise Exception("Invalid distribution ==> {}".format(latent_distrib))
        #self.decoder = Decoder3Fixed(output_dim=wd_embed_dim, n_latent=n_latent, sent_size = max_sent_len,
        #                              num_filters=num_filters, batch_size=batch_size)
            self.decoder = TransformerDecoder(wd_embed_dim=wd_embed_dim, num_units=num_units,
                                              n_layers=transformer_layers, n_latent=n_latent, sent_size = max_sent_len,
                                              batch_size = batch_size, ctx = ctx)
            self.vocab_size = self.bert.word_embed[0].params.get('weight').shape[0]
            self.out_embedding = gluon.nn.Embedding(input_dim=self.vocab_size, output_dim=wd_embed_dim, weight_initializer=mx.init.Uniform(0.1))
            self.inv_embed = InverseEmbed(batch_size, max_sent_len, self.wd_embed_dim, temp=wd_temp, ctx=self.model_ctx, params = self.out_embedding.params)
            self.ce_loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=-1, from_logits=True)

    def __call__(self, wp_toks, tok_types, valid_length=None):
        return super(BertTransVAE, self).__call__(wp_toks, tok_types, valid_length)

    def set_kl_weight(self, epoch, max_epochs):
        burn_in = int(max_epochs / 10)
        eps = 1e-6
        if epoch > burn_in:
            self.kld_wt = ((epoch - burn_in) / (max_epochs - burn_in)) + eps
        else:
            self.kld_wt = eps
        return self.kld_wt


    def forward(self, wp_toks, tok_types, valid_length=None):
        _, pooler_out_bert = self.bert(wp_toks, tok_types, valid_length)
        
        z, KL = self.latent_dist(pooler_out_bert, self.batch_size)
        y = self.decoder(z)
        
        #y_norm = mx.nd.norm(y, axis=-1, keepdims=True)   # so we can normalize by this norm
        #rec_y_1 = mx.nd.broadcast_div(y, y_norm) ## y / y_norm
        #rec_y = mx.nd.reshape(rec_y_1, (self.batch_size, self.max_sent_len, self.wd_embed_dim))
        
        ## does a matrix mult: rec_y = (32, 64, 300), shape mm = (32, 300, 25002)
        prob_logits = self.inv_embed(y)
        log_prob = mx.nd.log_softmax(prob_logits)
        recon_loss = self.ce_loss_fn(log_prob, wp_toks)
        kl_loss = (KL * self.kld_wt)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss, log_prob
        


class Decoder3Fixed(HybridBlock):

    def __init__(self, output_dim, n_latent=256, sent_size = 42, filter_size=4, num_filters=64, batch_size=8,
                 embed_dim=1024,
                 activation='relu'):
        super(Decoder3Fixed, self).__init__()
        
        t1 = sent_size + 2 * (filter_size - 1)
        t2 = int(math.floor((t1 - filter_size) / 2) + 1)
        t3 = int(math.floor((t2 - filter_size) / 2) + 1) - 2
        self._batch_size = batch_size
        self._n_latent = n_latent
        self._sent_size = sent_size
        self.dec_layers = nn.HybridSequential()
        with self.dec_layers.name_scope():
            self.dec_layers.add(
                gluon.nn.Conv2DTranspose(num_filters*2, (t3,1), strides=(2,1), output_padding=(1,0),
                                   in_channels=n_latent, activation=None))
            self.dec_layers.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True, in_channels=num_filters*2))
            self.dec_layers.add(gluon.nn.Activation(activation='relu'))
            self.dec_layers.add(
                gluon.nn.Conv2DTranspose(num_filters, (filter_size,1), strides=(2,1), 
                                   in_channels=num_filters*2, activation=None))
            self.dec_layers.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True, in_channels=num_filters))
            self.dec_layers.add(gluon.nn.Activation(activation='relu'))
            self.dec_layers.add(
                gluon.nn.Conv2DTranspose(1, (filter_size,output_dim), strides=(2,1),
                                   in_channels=num_filters, activation='relu'))


                    
    def __call__(self, x):
        return super(Decoder3Fixed, self).__call__(x)

    def hybrid_forward(self, F, x):
        x = F.reshape(x, (self._batch_size, self._n_latent, 1, 1)) ## back to rank 4 tensor for conv operations
        return self.dec_layers(x)


class TransformerDecoder(HybridBlock):
    def __init__(self, wd_embed_dim, num_units, num_heads=4, n_layers=6, n_latent=256, sent_size = 30, batch_size=8, ctx=mx.cpu()):
        super(TransformerDecoder, self).__init__()
        self._batch_size = batch_size
        self._sent_size = sent_size
        self._n_latent = n_latent
        self._wd_embed_dim = wd_embed_dim
        self._num_units = num_units
        with self.name_scope():
            self.projection = nn.Dense(in_units = n_latent, units = num_units)
            self.trans_block = TransformerBlock(
                attention_cell = 'multi_head',
                num_layers = n_layers,
                units = num_units,  
                hidden_size = 512,
                max_length = sent_size,
                num_heads = num_heads,
                scaled = True,
                dropout = 0.0,
                use_residual=True, output_attention=False,
                ctx = ctx)
            self.out_projection = nn.Dense(in_units = num_units, units = wd_embed_dim, flatten=False)

    def __call__(self, x):
        return super(TransformerDecoder, self).__call__(x)


    def hybrid_forward(self, F, x):
        ## x is shape (N, n_latent)
        x = self.projection(x)  ## Map n_latent ==> wd_embed_dim
        x = F.expand_dims(x, 1) ## (N, 1, wd_embed_dim)
        x = F.broadcast_to(x, (self._batch_size, self._sent_size, self._num_units))
        y, _ = self.trans_block(x)
        yp = self.out_projection(y)
        return yp


class TransformerEncoder(HybridBlock):
    def __init__(self, wd_embed_dim, num_units, num_heads=4, n_layers=6, n_latent=256, sent_size = 30, batch_size=8, ctx=mx.cpu()):
        super(TransformerEncoder, self).__init__()
        self._batch_size = batch_size
        self._sent_size = sent_size
        self._n_latent = n_latent
        with self.name_scope():
            self.in_projection = nn.Dense(in_units = wd_embed_dim, units = num_units, flatten=False)
            self.trans_block = TransformerBlock(
                attention_cell = 'multi_head',
                num_layers = n_layers,
                units = num_units,  
                hidden_size = 512,
                max_length = sent_size,
                num_heads = num_heads,
                scaled = True,
                dropout = 0.0,
                use_residual=True, output_attention=False,
                ctx = ctx)
            self.projection = nn.Dense(in_units = num_units, units = n_latent)

    def __call__(self, x):
        return super(TransformerEncoder, self).__call__(x)


    def hybrid_forward(self, F, x):
        ## x is shape (N, sent_size, wd_embed_dim)
        x = self.in_projection(x)
        y, _ = self.trans_block(x)
        first = y[:,0,:]
        encoding = self.projection(first)
        return encoding
    

def _position_encoding_init(max_length, dim):
    """Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc

def _get_layer_norm(use_bert, units):
    return nn.LayerNorm(in_channels=units)
    
    
class TransformerBlock(HybridBlock):
    """Transformer Encoder used as Decoder for BERT => Transformer Encoder/Decoder

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    use_residual : bool
    output_attention: bool, default False
        Whether to output the attention weights
    output_all_encodings: bool, default False
        Whether to output encodings of all encoder's cells, or only the last one
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    positional_weight: str, default 'sinusoidal'
        Type of positional embedding. Can be 'sinusoidal', 'learned'.
        If set to 'sinusoidal', the embedding is initialized as sinusoidal values and keep constant.
    use_bert_encoder : bool, default False
        Whether to use BERTEncoderCell and BERTLayerNorm. Set to True for pre-trained BERT model
    use_layer_norm_before_dropout: bool, default False
        Before passing embeddings to attention cells, whether to perform `layernorm -> dropout` or
        `dropout -> layernorm`. Set to True for pre-trained BERT models.
    scale_embed : bool, default True
        Scale the input embeddings by sqrt(embed_size). Set to False for pre-trained BERT models.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros',
                 positional_weight='sinusoidal', use_bert_encoder=False,
                 use_layer_norm_before_dropout=False, scale_embed=True, ctx=mx.cpu(),
                 prefix=None, params=None):
        super(TransformerBlock, self).__init__(prefix=prefix, params=params)
        print("active = {}".format(self._active))
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._ctx = ctx
        self._num_layers = num_layers
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        self._use_layer_norm_before_dropout = use_layer_norm_before_dropout
        self._scale_embed = scale_embed
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = _get_layer_norm(use_bert_encoder, units)
            self.position_weight = self._get_positional(positional_weight, max_length, units,
                                                        weight_initializer)
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                cell = self._get_encoder_cell(use_bert_encoder, units, hidden_size, num_heads,
                                              attention_cell, weight_initializer, bias_initializer,
                                              dropout, use_residual, scaled, output_attention, i)
                self.transformer_cells.add(cell)

        
    def _get_positional(self, weight_type, max_length, units, initializer):
        if weight_type == 'sinusoidal':
            encoding = _position_encoding_init(max_length, units)
            position_weight = self.params.get_constant('const', encoding)
        elif weight_type == 'learned':
            position_weight = self.params.get('position_weight', shape=(max_length, units),
                                              init=initializer)
        else:
            raise ValueError('Unexpected value for argument position_weight: %s'%(position_weight))
        return position_weight

    def _get_encoder_cell(self, use_bert, units, hidden_size, num_heads, attention_cell,
                          weight_initializer, bias_initializer, dropout, use_residual,
                          scaled, output_attention, i):
        return TransformerEncoderCell(units=units, hidden_size=hidden_size,
                    num_heads=num_heads, attention_cell=attention_cell,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer,
                    dropout=dropout, use_residual=use_residual,
                    scaled=scaled, output_attention=output_attention,
                    prefix='transformer%d_'%i)

    def __call__(self, inputs, states=None): #pylint: disable=arguments-differ
        return super(TransformerBlock, self).__call__(inputs, states)
    

    def forward(self, inputs, states=None): # pylint: disable=arguments-differ
        length = inputs.shape[1]
        if self._scale_embed:
            inputs = inputs * math.sqrt(inputs.shape[-1])
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        step_output, additional_outputs = super(TransformerBlock, self).forward(inputs, states)
        return step_output, additional_outputs


    def hybrid_forward(self, F, inputs, states=None, position_weight=None):
        # pylint: disable=arguments-differ
        if states is not None:
            steps = states[-1]
            positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
            inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=0))
        if self._use_layer_norm_before_dropout:
            inputs = self.layer_norm(inputs)
            inputs = self.dropout_layer(inputs)
        else:
            inputs = self.dropout_layer(inputs)
            inputs = self.layer_norm(inputs)
        outputs = inputs

        all_encodings_outputs = []
        additional_outputs = []
        batch_size = inputs.shape[0]
        
        for i,cell in enumerate(self.transformer_cells):
            outputs, attention_weights = cell(inputs, None)
            inputs = outputs
            if self._output_all_encodings:
                all_encodings_outputs.append(outputs)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if self._output_all_encodings:
            return all_encodings_outputs, additional_outputs
        else:
            return outputs, additional_outputs
    

    
