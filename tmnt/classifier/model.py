# codeing: utf-8

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import HybridBlock
import gluonnlp as nlp


class DANVAETextClassifier(HybridBlock):
    def __init__(self, vae_model, emb_input_dim, emb_output_dim, dropout=0.2, dense_units=[100,100], emb_dropout=0.5,
                 seq_length = 64, num_classes=2, non_vae_weight=1.0):

        super(DANVAETextClassifier, self).__init__()

        self.seq_length = seq_length
        self.emb_dim = emb_output_dim
        self.non_vae_weight = non_vae_weight
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            self.pooler = gluon.nn.AvgPool2D((self.seq_length,1)) ## average pool over time/sequence
            self.vae_encoder = gluon.nn.HybridSequential()
            self.emb_drop = gluon.nn.Dropout(emb_dropout)
            self.encoder = gluon.nn.HybridSequential()
            with self.encoder.name_scope():
                for u in dense_units:
                    self.encoder.add(gluon.nn.Dropout(dropout))
                    self.encoder.add(gluon.nn.Dense(units=u, use_bias=True, activation='relu'))
            with self.vae_encoder.name_scope():
                self.vae_encoder.add(vae_model.embedding)
                self.vae_encoder.add(vae_model.encoder)
                self.vae_encoder.add(vae_model.latent_dist.mu_encoder)
                self.vae_encoder.add(gluon.nn.Dropout(dropout))
            self.output = gluon.nn.HybridSequential()
            self.output.add(gluon.nn.Dropout(dropout))
            self.output.add(gluon.nn.Dense(in_units=20, units=num_classes, use_bias=True))

    def hybrid_forward(self, F, bow_data, data, mask):
        embedded = self.embedding(data)
        mask = F.expand_dims(mask, axis=2)
        embedded = embedded * mask
        embedded = self.emb_drop(embedded)
        pooled = self.pooler(F.reshape(embedded, (-1,1,self.seq_length, self.emb_dim)))
        encoded = self.encoder(pooled) * self.non_vae_weight
        encoded2 = F.concat(encoded, self.vae_encoder(bow_data), dim=1)
        return self.output(encoded2)
        


class DANTextClassifier(HybridBlock):
    def __init__(self, emb_input_dim, emb_output_dim, dropout=0.2, dense_units=[100,100], emb_dropout=0.5,
                 seq_length = 64, n_classes=2):
        super(DANTextClassifier, self).__init__()

        self.seq_length = seq_length
        self.emb_dim = emb_output_dim
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            self.pooler = gluon.nn.AvgPool2D((self.seq_length,1)) ## average pool over time/sequence
            self.encoder = gluon.nn.HybridSequential()
            self.emb_drop = gluon.nn.Dropout(emb_dropout)
            with self.encoder.name_scope():
                for u in dense_units:
                    self.encoder.add(gluon.nn.Dropout(dropout))
                    self.encoder.add(gluon.nn.Dense(units=u, use_bias=True, activation='relu'))
            self.output = gluon.nn.HybridSequential()
            self.output.add(gluon.nn.Dropout(dropout))
            self.output.add(gluon.nn.Dense(in_units = dense_units[-1], units=n_classes, use_bias=True))

    def hybrid_forward(self, F, bow_data, data, mask):
        embedded = self.embedding(data)
        mask = F.expand_dims(mask, axis=2)
        embedded = embedded * mask
        embedded = self.emb_drop(embedded)
        pooled = self.pooler(F.reshape(embedded, (-1,1,self.seq_length, self.emb_dim)))
        encoded = self.encoder(pooled)
        return self.output(encoded)


    
