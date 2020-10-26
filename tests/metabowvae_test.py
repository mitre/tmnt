import numpy as np
import gluonnlp as nlp
from scipy.sparse import csr_matrix
from tmnt.models.bow.bow_vae import MetaBowVAE

X_scipy = csr_matrix(np.ones((100, 100)))
y_numpy = np.ones((100,1))
vocabulary = nlp.Vocab(nlp.data.Counter(['a'*i for i in range(100)]), unknown_token=None, padding_token=None, bos_token=None, eos_token=None)

def test_train_and_perplexity_scalar():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    model.perplexity(X_scipy, y_numpy)
    assert(True)

def test_train_and_perplexity_categorical():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    model.perplexity(X_scipy, y_numpy)
    assert(True)

def test_train_and_npmi_scalar():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    assert(model._npmi_per_covariate(X_scipy, y_numpy, 10) == 0)

def test_train_and_npmi_categorical():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    assert(model._npmi_per_covariate(X_scipy, y_numpy, 10) == 0)

def test_train_and_transform_scalar():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    trans = model.transform(X_scipy, y_numpy)
    assert(np.all(trans == trans[0]))

def test_train_and_transform_categorical():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    trans = model.transform(X_scipy, y_numpy)
    assert(np.all(trans == trans[0]))

def test_train_and_topics_scalar():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    model.get_topic_vectors()
    assert(True)

def test_train_and_topics_categorical():
    model = MetaBowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy, y_numpy)
    model.get_topic_vectors()
    assert(True)
