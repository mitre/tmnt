import numpy as np
from scipy.sparse import csr_matrix
from tmnt.models.bow.bow_vae import BowVAE
import gluonnlp as nlp

X_scipy = csr_matrix(np.ones((100,100)))
vocabulary = nlp.Vocab(nlp.data.Counter(['a'*i for i in range(100)]), unknown_token=None, padding_token=None, bos_token=None, eos_token=None)

def test_train_and_perplexity_scipy():
    model = BowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy)
    model.perplexity(X_scipy)
    assert(True)

def test_train_and_npmi_scipy():
    model = BowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy)
    assert(model.npmi(X_scipy, 10)[0] == 0)

def test_train_and_transform_scipy():
    model = BowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy)
    enc = model.transform(X_scipy)
    assert(np.all(enc[0] == enc))

def test_train_and_get_topics_scipy():
    model = BowVAE(vocabulary, batch_size=32)
    model.fit(X_scipy)
    model.get_topic_vectors()
    assert(True)
