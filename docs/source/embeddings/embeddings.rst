Training Custom Word Embeddings
===============================

Given TMNT's use of word embeddings to (optionally) initialize the embedding
layer of the topic encoder, in some cases, especially for data from specialized
domains (or other languages), training custom word embeddings on relevant
datasets can help provide higher quality topic models. In addition, some
advances features of TMNT, such as incorporating a coherence loss term,
rely on quality word embeddings that capture word similarity accurately.

While many existing frameworks, including the original ``word2vec`` implementation,
are available for training word embeddings, we provide functionality to
do so within TMNT for convenience.  This implementation builds on the existing
mechanisms for training word embeddings provided by the GluonNLP library.

Embeddings are trained using the ``bin/train_embeddings.py`` script with the
following command-line options.

===================  ============    =================================================================
Argument             Type            Description
===================  ============    =================================================================
data_type            string          Corpus type (twitter, news, etc.) - triggers appropriate preprocessing
data_root            string          File path to root of directory structure containing text corpus
file_pattern         string          Regular expression to match files in directory structure (default: *.txt)
batch_size           integer         Batch size for training. (default: 1000)
epochs               integer         Number of passes over the entire training set (default: 5)
gpu                  integer         Device ID for GPU to use; CPU if None (default: None)
no_prefetch_batch    flag            Disables multi-threaded pre-fecthing of batches (default: False)
num_prefetch_epoch   integer         Start data pipeline for N epochs when beginning current epoch (default: 3)
no_hybridize         flag            Disable hybridization (default: False)
pre_embedding_name   string          Use an existing GluonNLP pre-trained embedding as a starting point (default: None)
emsize               integer         Size of embedding dimension (when not using pre_embedding)
ngrams               integer list    Character n-grams sizes to use (for character-level embedding models)
ngram_buckets        integer         Number of n-gram buckets/hashes to use (set to 0 for standard word2vec embeddings)
model                string          SkipGram or cbow (default: skipgram)
window               integer         Context size in terms of tokens/words to left/right
negative             integer         Number of negative samples to use for each source/context word
max_vocab_size       integer         Limit words considered (by frequency); OOV words ignored
model_export         string          Export model in standard word2vec format to specified file
token_embedding      string          Export model in GluonNLP format to specifed file
optimizer            string          MXNet optimizer to use (e.g. sgd, adam)
lr                   float           Learning rate
seed                 integer         Random seed
logdir               string          Directory for storing logs (default: logs)
log_interval         integer         Frequency with which to report training information
eval_interval        integer         Frequency with which to evaluate model
===================  ============    =================================================================
