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
TMNT can be used to train a transformer-based VAE (using BERT base pre-trained as the encoder), with
an invocation such as::

  python bin/seq_vae.py --input_file hotel_reviews.900.txt --gpus 0 --gen_lr 1e-3 --min_lr 0.000001 \\
  --batch_size 300 --sent_size 42 --epochs 200 --latent_dim 50 --log_interval 3 --transformer_layers 6 \\
  --save_dir ./_exps --kld_wt 0.01 --latent_dist vmf --embedding_source 'glove.6B.300d' --optimizer adam --num_units 512

The input file should consist of a simple text file with each individual sentence/passage of text on a separate line.
Logging output will appear in a sub-directory inside the directory path specified by ``save_dir``.

The options for ``seq_vae.py`` are described below

===================  ===========    =================================================================
Option               Type           Description
===================  ===========    =================================================================
input_file           string         Path to input file (single sequence/sentence per line)
batch_size           integer        Batch size (usually ranges between 32 to 64; larger is better)
epochs               integer        Number of passes over entire training set
optimizer            string         Optimizer (adam, sgd, bertadam)
gpus                 integer        Only single GPU supported currently
gen_lr               float          Maximum learning rate (achieved after warmup of 10% of total updates)
min_lr               float          Minimum learning rate (after decay, reached at 80% of total updates)
offset_factor        float          Shifts the cosine schedule/phase left/right for learning rate
sent_size            integer        Maximum sentence/sequence lengths for inputs
num_units            integer        Number of units in transformer blocks
transformer_layers   integer        Number of transformer layers in the decoder and encoder (if not BERT)
wd_embed_dim         integer        Number of units in output embedding (for BERT encoder)
latent_dim           integer        Dimensionality of the latent distribution
latent_dist          string         Latent distribution type (``gaussian``, ``logistic_gaussian``, ``vmf``, ``gaussian_unitvar``)
kappa                float          Concentration parameter when using von Mises Fisher (vmf) distribution
kld_wt               float          Coefficient to weight the KL divergence loss term (usually ranges from 0.1 to 1)
log_interval         integer        Number of batches to process before outputing loss (and example reconstruction)
save_model_freq      integer        Number of epochs between each model save
weight_decay         float          Standard weight decay for optimizer
warmup_ratio         float          Percentage of training steps after which decay begins (default 0.1)
use_bert             boolean        Use BERT (base) as the encoder instead of non-pre-trained transformer
embedding_source     string         GluonNLP embedding source to use (when not using BERT encoder)
