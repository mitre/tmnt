"""
Training a seq2bow encoder-decoder model
========================================
"""
from tmnt.estimator import SeqBowEstimator
import numpy as np
import os
import logging
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.distribution import LogisticGaussianDistribution
from tmnt.utils.log_utils import logging_config
from tmnt.data_loading import get_llm_dataloader

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)

tr_size = 100
train_data = data[:tr_size]
dev_data   = data[-tr_size:]
train_y    = y[:tr_size]
dev_y      = y[-tr_size:]
#model_name = 'bert_12_768_12'
#dataset = 'book_corpus_wiki_en_uncased'
batch_size = 32
seq_len = 64
pad = True

vectorizer = TMNTVectorizer(vocab_size=2000)
vectorizer.fit_transform(train_data)

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

train_y_s = ['class_'+str(y) for y in train_y]
dev_y_s = ['class_'+str(y) for y in dev_y]

#tr_ds = ArrayDataset(train_data, train_y_s)
#dev_ds = ArrayDataset(dev_data, dev_y_s)

print("Classes = {}".format(classes))

tf_llm_name = 'distilbert-base-uncased'

train_ds = list(zip(train_y_s, train_data))
dev_ds   = list(zip(dev_y_s, dev_data))
aux_ds   = None
label_map = { l:i for i,l in enumerate(classes) }
train_loader = get_llm_dataloader(train_ds, vectorizer, tf_llm_name, label_map, 16, 128 )
dev_loader = get_llm_dataloader(dev_ds, vectorizer, tf_llm_name, label_map, 16, 128)

latent_distribution = LogisticGaussianDistribution(768,80,dr=0.1,alpha=2.0)

estimator = SeqBowEstimator(llm_model_name = tf_llm_name,
                            latent_distribution = latent_distribution,
                            n_labels = num_classes,
                            bow_vocab = vectorizer.get_vocab(),
                            optimizer='bertadam',
                            batch_size=batch_size, device='cpu', log_interval=1,
                            log_method=log_method, gamma=100.0, 
                            lr=2e-5, decoder_lr=0.01, epochs=20)


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


