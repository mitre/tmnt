"""
Training a seq2bow encoder-decoder model
========================================
"""
from tmnt.estimator import SeqBowMetricEstimator
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.distribution import LogisticGaussianDistribution
from tmnt.utils.log_utils import logging_config
from tmnt.data_loading import get_llm_paired_dataloader
import torch

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)

tr_size = 100
train_data_a = data[:tr_size]
dev_data_a   = data[-tr_size:]
train_y_a    = y[:tr_size]
dev_y_a      = y[-tr_size:]
train_data_b = data[tr_size:(tr_size * 2)]
dev_data_b   = data[-(tr_size*2):-tr_size]
train_y_b    = y[tr_size:(tr_size * 2)]
dev_y_b      = y[-(tr_size*2):-tr_size]
batch_size = 32
seq_len = 64
pad = True

vectorizer = TMNTVectorizer(vocab_size=2000)
vectorizer.fit_transform(train_data_a)

supervised  = True
use_logging = True

if supervised:
    num_classes = int(np.max(y) + 1)
    classes = ['class_'+str(i) for i in range(num_classes)]
else:
    num_classes = 0
    classes = None

if use_logging:    
    logging_config(folder='.', name='f_seqbow_20news', level='info', console_level='info')
    log_method = 'log'
else:
    log_method = 'print'

train_y_a_s = ['class_'+str(y) for y in train_y_a]
dev_y_a_s = ['class_'+str(y) for y in dev_y_a]

train_y_b_s = ['class_'+str(y) for y in train_y_b]
dev_y_b_s = ['class_'+str(y) for y in dev_y_b]

print("Classes = {}".format(classes))

tf_llm_name = 'distilbert-base-uncased'

train_ds_a = list(zip(train_y_a_s, train_data_a))
dev_ds_a   = list(zip(dev_y_a_s, dev_data_a))
train_ds_b = list(zip(train_y_b_s, train_data_b))
dev_ds_b   = list(zip(dev_y_b_s, dev_data_b))
aux_ds   = None
label_map = { l:i for i,l in enumerate(classes) }
train_loader = get_llm_paired_dataloader(train_ds_a, train_ds_b, vectorizer, tf_llm_name, label_map, 10, 10, 128 )
dev_loader = get_llm_paired_dataloader(dev_ds_a, dev_ds_b, vectorizer, tf_llm_name, label_map, 10, 10, 128)

latent_distribution = LogisticGaussianDistribution(768,80,dr=0.1,alpha=2.0)

device = torch.device('cpu')

estimator = SeqBowMetricEstimator(llm_model_name = tf_llm_name,
                            latent_distribution = latent_distribution,
                            n_labels = num_classes,
                            bow_vocab = vectorizer.get_vocab(),
                            optimizer='bertadam',
                            batch_size=batch_size, device=device, log_interval=1,
                            log_method=log_method, gamma=10.0, 
                            lr=2e-5, decoder_lr=0.0004, epochs=20)


# this will take quite some time without a GPU!
#estimator.fit_with_validation(train_loader, dev_loader, None)
#os.makedirs('_model_dir', exist_ok=True)

# save the model
#estimator.write_model('./_model_dir')

# %%
# An inference object is then created which enables the application of the model to raw text
# data and/or directly to document-term matrices
#from tmnt.inference import SeqVEDInferencer
#inferencer = SeqVEDInferencer(estimator, max_length=seq_len, pre_vectorizer=vectorizer, ctx=ctx)


# %%
# We can visualize the topics and associated topic terms using PyLDAvis. Note, this simply shows the mechanics
# for this step; reasonable topic models will require additional training on the entire dataset.
#import pyLDAvis
#import funcy
#full_model_dict = inferencer.get_pyldavis_details(tr_dataset)
#pylda_opts = funcy.merge(full_model_dict, {'mds': 'mmds'})
#vis_data = pyLDAvis.prepare(**pylda_opts)

# %%
# The topic model terms and topic-term proportions will be written
# to the file ``m1.html``
#pyLDAvis.save_html(vis_data, 'm1.html')


