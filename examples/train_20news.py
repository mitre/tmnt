from tmnt.estimator import BowEstimator
import gluonnlp as nlp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from tmnt.inference import BowVAEInferencer


n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
data_samples = data[:2000]

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(data_samples)
vv = {v: 1 for v in tf_vectorizer.vocabulary_}
vocab = nlp.Vocab(vv, unknown_token=None, padding_token=None, bos_token=None, eos_token=None)
estimator = BowEstimator(vocab)
_ = estimator.fit(tf)
inferencer = BowVAEInferencer(estimator.model)
inferencer.get_top_k_words_per_topic(5)


