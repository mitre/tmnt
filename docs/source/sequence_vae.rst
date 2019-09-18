Sequence Variational AutoEncoders
=================================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:


1. Overview and Background
++++++++++++++++++++++++++

This section describes some advanced uses of TMNT for building different types of variational
autoencoders over input word *sequences*.  These models may be more useful for creating
topic (or *motif*) models over shorter pieces of text.  In addition, they have other interesting
properties when used *generatively*.  The work here is based on the ideas found in Zhang et al.:

https://arxiv.org/abs/1708.04729

While implementations are available for convolutional VAEs following Zhang et al., we have
found that VAEs using the transformer architecture may perform better and provide greater flexibility
for variable-length inputs.

2. Training a Transformer-based VAE: Quick Start
++++++++++++++++++++++++++++++++++++++++++++++++

Assuming TMNT has been installed and tested following the instructions in Installation and Quick Start,
TMNT can be used to train a transformer-based VAE, with
an invocation such as::

  mkdir _exps 
  mkdir _model_seq_vae
  python ~/tmnt/bin/seq_vae.py --input_file ./data/seq_vae/hotel_reviews.900.txt \
  --gpus 0 --gen_lr 1e-3 --min_lr 0.00002 --batch_size 450 --sent_size 64 --epochs 600 \
  --latent_dim 50 --log_interval 2 --transformer_layers 4 --save_dir ./_exps/ --kld_wt 0.1 \
  --latent_dist vmf --optimizer adam --num_units 512 --kappa 50.0 --num_heads 4 \
  --model_dir _model_seq_vae


In contrast to the bag-of-words VAE, it is strongly recommended to use GPUs to accelerate training with batch
sizes as large as memory will allow.

Note that the exact options to use will vary considerably from dataset to dataset.  The above invocation
will converge (and likely overfit) on the small sample datset of 900 text inputs.  For larger training
sets, reduce the number of epochs.  Learning rates may need to be tuned.  The ``kappa`` parameter
controls the concentration (or variance) for the ``vmf`` distribution; lower values will
prevent overfitting.  

The input file should consist of a simple text file with each individual sentence/passage of text on a separate line.
Logging output will appear in a sub-directory inside the directory path specified by ``save_dir``.
An alternative input is a json list where a text field is encoded as json string with a key
specified by the option ``json_text_key``.

The options for ``seq_vae.py`` are described below.

===================  ===========    =================================================================================
Option               Type           Description
===================  ===========    =================================================================================
input_file           string         Path to input file (single sequence/sentence per line)
batch_size           integer        Batch size (usually ranges between 32 to 64; larger is better)
epochs               integer        Number of passes over entire training set
optimizer            string         Optimizer (adam, sgd, bertadam)
gpus                 integer        Only single GPU supported currently
gen_lr               float          Maximum learning rate (achieved after warmup of 10% of total updates)
min_lr               float          Minimum learning rate (after decay, reached at 80% of total updates)
offset_factor        float          Shifts the cosine schedule/phase left/right for learning rate (default 1.0 is usually fine)
sent_size            integer        Maximum sentence/sequence lengths for inputs
num_units            integer        Number of units in transformer blocks
num_heads            integer        Number of heads in transformer self-attention
hidden_size          integer        Size of the hidden dimensions in Transformer blocks
transformer_layers   integer        Number of transformer layers in the decoder and encoder (if not BERT)
wd_embed_dim         integer        Number of units in output embedding (for BERT encoder only)
latent_dim           integer        Dimensionality of the latent distribution
latent_dist          string         Latent distribution type (``gaussian``, ``logistic_gaussian``, ``vmf``, ``none``)
kappa                float          Concentration parameter when using ``vmf`` distribution; lower values increase regularization
kld_wt               float          Coefficient to weight the KL divergence loss term
log_interval         integer        Number of batches to process before outputing loss (and example reconstruction)
save_dir             string         Directory where experiment log will be saved
model_dir            string         Model directory to save final model parameters
weight_decay         float          Standard weight decay for optimizer
warmup_ratio         float          Percentage of training steps after which decay begins (default 0.1)
use_bert             boolean        Use BERT (base) as the encoder instead of non-pre-trained transformer
embedding_source     string         GluonNLP embedding source to use (when not using BERT encoder)
json_text_key        string         String key for a json text field to use as text input
label_smoothing      float          Standard label smoothing parameter (NOT YET SUPPORTED)
===================  ===========    =================================================================================


3. Sampling from a Trained Model
++++++++++++++++++++++++++++++++

Assuming TMNT has been installed a model has been trained (i.e. the files ``model.config``, ``model.params`` and
``vocab.json`` exist), one can generate sampled reconstructions of an input text to test the model as follows using
the method ``recode_text`` as follows::

  >>> from tmnt.seq_vae.runtime import SeqVAEInference
  >>> infer = SeqVAEInference('model.params', 'model.config', 'vocab.json')
  >>> infer.recode_text('just will let you know that our recent trip to la to be on the tv show " dancing with the stars " was made complete by our outstanding 3 night stay at the custom hotel . nothing was very difficult for the staff .')

The method will return a list of tokens representing the reconstructed text by running the text through the
full variational autoencoder, *including the stochastic step of drawing a sample from the latent distribution
before decoding the encoded input*.  This means the method is non-deterministic and may return different
results on each invocation.
