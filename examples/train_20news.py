from tmnt.estimator import BowEstimator
import gluonnlp as nlp
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import BowVAEInferencer


n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
data_samples = data[:2000]

tf_vectorizer = TMNTVectorizer(vocab_size=1000)

X, _ = tf_vectorizer.fit_transform(data_samples)
estimator = BowEstimator(tf_vectorizer.get_vocab()).fit(X)
inferencer = BowVAEInferencer(estimator.model)
encodings = inferencer.encode_texts(['Greater Armenia would stretch from Karabakh, to the Black Sea, to the Mediterranean, so if you use the term Greater Armenia use it with care.','I have two pairs of headphones I\'d like to sell.  These are excellent, and both in great condition'])




