TMNT Summary of Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TMNT provides a number of latent variable models useful for text analytics, including
topic models. These models are often useful in situations where there is a need
to characterize and understand the content in a large text corpus, typically in
situations without any particular a priori information need.

Fully unsupervised latent variable models for text are learned through a
**variational auto-encoder** (VAE) architecture.  VAEs consist of an *encoder*
that learns to map inputs to a latent distribution and a *decoder* which learns
to reconstruct the original text input from samples drawn from the latent distribution
provided by the encoder. 

When supervisory information is available in the form of document *labels*, TMNT is
able to jointly train a latent variable topic model together with a **text classifier**.
In situations when supervisory information exists, such labels are available for only
a subset of a much larger dataset. In such cases, TMNT provides methods for
**semi-supervised learning** whereby a text classifier is jointly learned with a topic model;

Recent advances in **metric learning** faciliate few shot learning and better generalization
in certain domains. TMNT implements joint metric learning and topic modeling; as with
standard supervised learning, semi-supervised extensions are possible that leverage
mixtures of labeled/linked data (for metric learning) and unlabeled data (for topic modeling).

Standard unigram (bag-of-words) or n-gram document representations are used in TMNT as is
commonplace with topic models and many text classifiers. In the simplest instances, the
encoder and decoder are both realized as one or more fully connected layers. TMNT also provides
a richer form of topic model trained as a variational encoder-decoder model where
the encoder consists of pre-trained **BERT** models. BERT-based encoders may be used
with fully unsuperivsed topic models as well as with fully supervised and semi-supervised
classifcation and metric learning.


TMNT inlcludes some additional capabilities, including:

 * Use of AutoGluon for hyperparameter and architecture optimization
 * Multiple latent distributions, including the von Mises Fisher distribution 
 * Use of the ``PyLDAvis`` library to visualize learned topics
 * UMAP to learn and visualize topic embeddings
 * A robust text pre-processor wrapping the sklearn ``CountVectorizer`` class.   
 * Runtime/inference API to allow for easy deployment of learned topic models
