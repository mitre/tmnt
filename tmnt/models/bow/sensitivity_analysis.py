# coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""

import mxnet as mx
import numpy as np
import time
import logging
from tmnt.models.bow.bow_models import BowNTM, MetaDataBowNTM
from tmnt.models.bow.runtime import BowNTMInference
from tmnt.models.bow.bow_doc_loader import DataIterLoader, file_to_sp_vec

__all__ = ['get_encoder_jacobians_at_data_file', 'get_jacobians_at_data_file']

def _get_sampled_topic_embeddings(model, data, covars, batch_size):
    emb_out = model.embedding(data)
    co_emb = mx.nd.concat(emb_out, covars)
    z, KL = model.run_encode(mx.nd.ndarray, co_emb, batch_size)
    return z

def _get_encoding(model, data, covars, batch_size):
    emb_out = model.embedding(data)
    co_emb = mx.nd.concat(emb_out, covars)
    return model.latent_dist.mu_encoder(model.encoder(co_emb))

def _get_jacobians_at_data(model, dataloader, f_covars, batch_size, sample_size):
    """
    For a given model, compute sum of jacobian matrix values at datapoint in dataloader
    """
    jacobians = mx.nd.zeros(shape=(model.vocab_size, len(f_covars), model.n_latent))
    f_covar_mapping = {}
    for i,c in enumerate(f_covars):
        f_covar_mapping[c] = i
    for bi, (data, covars) in enumerate(dataloader):
        if bi * batch_size >= sample_size:
            break
        covars = mx.nd.expand_dims(covars, axis=1)
        z_data = _get_sampled_topic_embeddings(model, data, covars, batch_size)
        for i in range(model.vocab_size):
            #if i > 0 and i % 100 == 0:
            #    print("... {} vocab items".format(i))
            z_data.attach_grad()
            with mx.autograd.record():
                yy = model.cov_decoder(z_data, covars)
                yi = yy[:, i] ## for the ith term, over batch
            yi.backward(retain_graph=True)
            cv_is = [f_covar_mapping[covars[j].asscalar()] for j in range(batch_size)]
            ## shape (N x C) - where is number of covariates            
            onehots = mx.nd.expand_dims(mx.nd.one_hot(mx.nd.array(cv_is), len(f_covars)), axis=2)
            ## shape (N x T) where T is number of topics
            zz = mx.nd.expand_dims(z_data.grad, axis=1)
            op = onehots * zz ## outer product
            jacobians[i] += op.sum(axis=0)
    return mx.nd.swapaxes(jacobians, 1, 0)


def get_jacobians_at_data_file(model, data_file, covars, sample_size=200):
    batch_size = 50
    data_mat, _, data_labels_list, _ = file_to_sp_vec(data_file, model.vocab_size, scalar_labels=True, encoding='utf-8')
    data_labels = mx.nd.array(data_labels_list, dtype='float32')
    data_labels = (data_labels - data_labels.min()) / (data_labels.max() - data_labels.min()) ## normalize
    dataloader = DataIterLoader(mx.io.NDArrayIter(data_mat, data_labels, batch_size, last_batch_handle='discard', shuffle=True))
    return _get_jacobians_at_data(model, dataloader, covars, batch_size, sample_size)


def _get_encoder_jacobians_at_data(model, dataloader, batch_size, sample_size):
    """
    For a given model, compute sum of jacobian matrix values at datapoint in dataloader
    """
    jacobians = mx.nd.zeros(shape=(model.n_latent, model.vocab_size))
    #jacobians = mx.nd.zeros(shape=(model.n_latent, 200))
    for bi, (data, covars) in enumerate(dataloader):
        if bi * batch_size >= sample_size:
            print("Sample processed, exiting..")
            break
        covars = mx.nd.expand_dims(covars, axis=1)
        x_data = data.tostype('default')
        for i in range(model.n_latent):
            x_data.attach_grad()
            with mx.autograd.record():
                emb_out = model.embedding(x_data)
                co_emb = mx.nd.concat(emb_out, covars)
                yi = co_emb[:, i] ## for the ith topic, over batch
            yi.backward()
            ss = x_data.grad.sum(axis=0)
            jacobians[i] += ss
        print("Batch {} totally processed".format(bi))
    return jacobians


def get_encoder_jacobians_at_data_nocovar(model, dataloader, batch_size, sample_size, ctx):
    """
    For a given model, compute sum of jacobian matrix values at datapoint in dataloader
    """
    jacobians = np.zeros(shape=(model.n_latent, model.vocab_size))
    for bi, (data, _) in enumerate(dataloader):
        if bi * batch_size >= sample_size:
            print("Sample processed, exiting..")
            break
        x_data = data.tostype('default')
        x_data = x_data.as_in_context(ctx)
        for i in range(model.n_latent):
            x_data.attach_grad()
            with mx.autograd.record():
                emb_out = model.embedding(x_data)
                enc_out = model.latent_dist.mu_encoder(model.encoder(emb_out))
                yi = enc_out[:, i] ## for the ith topic, over batch
            yi.backward()
            mx.nd.waitall()
            ss = x_data.grad.sum(axis=0).asnumpy()
            jacobians[i] += ss
    return jacobians


def get_encoder_jacobians_at_data_file(model, data_file, sample_size=20000):
    batch_size = 1000
    data_mat, _, data_labels_list, _ = file_to_sp_vec(data_file, model.vocab_size, scalar_labels=True, encoding='utf-8')
    data_labels = mx.nd.array(data_labels_list, dtype='float32')
    data_labels = (data_labels - data_labels.min()) / (data_labels.max() - data_labels.min()) ## normalize
    dataloader = DataIterLoader(mx.io.NDArrayIter(data_mat, data_labels, batch_size, last_batch_handle='discard', shuffle=True))
    return get_encoder_jacobians_at_data_nocovar(model, dataloader, batch_size, sample_size, mx.cpu())
