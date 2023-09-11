"""
Training a seq2bow encoder-decoder model
========================================
"""
from tmnt.estimator import SeqBowMetricEstimator
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.distribution import LogisticGaussianDistribution, Projection, VonMisesDistribution
from tmnt.utils.log_utils import logging_config
from tmnt.data_loading import get_llm_paired_dataloader, get_llm_dataloader, StratifiedPairedLLMLoader
import torch

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)

tr_size = 1000
train_data_a = data[:tr_size]
dev_data_a   = data[-tr_size:]
train_y_a    = y[:tr_size]
dev_y_a      = y[-tr_size:]
train_data_b = data[tr_size:(tr_size * 2)]
dev_data_b   = data[-(tr_size*2):-tr_size]
train_y_b    = y[tr_size:(tr_size * 2)]
dev_y_b      = y[-(tr_size*2):-tr_size]
aux_data     = data[(tr_size *2): (tr_size * 4)]
batch_size = 20 

vectorizer = TMNTVectorizer(vocab_size=2000)
vectorizer.fit_transform(train_data_a + train_data_b)

use_logging = True

num_classes = int(np.max(y) + 1)
classes = ['class_'+str(i) for i in range(num_classes)]

if use_logging:    
    logging_config(folder='.', name='h_seqbow_20news', level='info', console_level='info')
    log_method = 'log'
else:
    log_method = 'print'

train_y_a_s = ['class_'+str(y) for y in train_y_a]
dev_y_a_s = ['class_'+str(y) for y in dev_y_a]

train_y_b_s = ['class_'+str(y) for y in train_y_b]
dev_y_b_s = ['class_'+str(y) for y in dev_y_b]

#tf_llm_name = 'distilbert-base-uncased'
tf_llm_name = 'sentence-transformers/all-mpnet-base-v2'

train_ds_a = list(zip(train_y_a_s, train_data_a))
dev_ds_a   = list(zip(dev_y_a_s, dev_data_a))
train_ds_b = list(zip(train_y_b_s, train_data_b))
dev_ds_b   = list(zip(dev_y_b_s, dev_data_b))
aux_ds     = list(zip([0] * len(aux_data), aux_data))

label_map = { l:i for i,l in enumerate(classes) }
device = torch.device('cpu')

#train_loader = get_llm_paired_dataloader(train_ds_a[:100], train_ds_a[:100], vectorizer, tf_llm_name, label_map, 20, 256, 256 , shuffle_both=True, device=device)
#train_loader = StratifiedPairedLLMLoader(train_ds_a, train_ds_b, vectorizer, tf_llm_name, label_map, 20, 512, 512, device=device)
train_loader = StratifiedPairedLLMLoader(train_ds_a, train_ds_b, vectorizer, tf_llm_name, label_map, 20, 128, 128, device=device)
#dev_loader = get_llm_paired_dataloader(dev_ds_a, dev_ds_b, vectorizer, tf_llm_name, label_map, 50, 256, 256, device=device)
dev_loader = StratifiedPairedLLMLoader(dev_ds_a, dev_ds_b, vectorizer, tf_llm_name, label_map, 20, 512, 512, device=device)
#aux_loader = get_llm_dataloader(aux_ds, vectorizer, tf_llm_name, label_map, 10, 128, shuffle=True, device=device) 

#latent_distribution = LogisticGaussianDistribution(768,10,dr=0.05,alpha=1.0,device=device)
#latent_distribution = VonMisesDistribution(768,10,dr=0.05,device=device)
latent_distribution = Projection(768,200,device=device)


estimator = SeqBowMetricEstimator(llm_model_name = tf_llm_name,
                                  vocabulary = vectorizer.get_vocab(),
                            latent_distribution = latent_distribution,
                                  sdml_smoothing_factor=0.0,
                            batch_size=batch_size, device=device, log_interval=1,
                            log_method=log_method, gamma=10000.0, entropy_loss_coef=1000.0, 
                            lr=1e-5, decoder_lr=0.001, epochs=1)


# this will take quite some time without a GPU!
#estimator.fit_with_validation(train_loader, dev_loader, None)
#os.makedirs('_model_dir', exist_ok=True)

#encodings = []
#for data in aux_loader:
#    _, input_ids1, mask1, bow1 = data
#    encodings.append(estimator.model.unpaired_input_encode(input_ids1, mask1))
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


