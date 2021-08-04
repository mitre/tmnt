Preprocessing
=============

For topic models that make use of bag-of-word (or bag-of-n-gram) representations, TMNT provides
some routines to help pre-process text data. These routines map text documents into (sparse) vectors
that represent the (possibly weighted) counts of terms that appear. The functionalitiy is
contained within the class :py:class:`tmnt.preprocess.vectorizer.TMNTVectorizer`.

Below provides a code snippet highlighting how to process a simple set of strings/documents::

  from tmnt.preprocess.vectorizer import TMNTVectorizer
  corpus = [
       'This is the first document.',
       'This document is the second document.',
       'And this is the third one.',
       'Is this the first document?',
   ]
  vectorizer = TMNTVectorizer()
  X, _ = vectorizer.fit_transform(corpus)
  print(X.toarray())
  print(vectorizer.get_vocab().token_to_idx)

The resulting document-term matrix `X` can be used for training a topic model.

``TMNTVectorizer`` simply wraps (rather than extends) :py:class:`sklearn.feature_extraction.text.CountVectorizer`
and provides some additional functionality useful for handling JSON list input representations.
When not using JSON input formats and/or working with purely unlabeled data, simply using
``CountVectorizer`` makes sense.  The above example is very similar::

  from sklearn.feature_extraction.text import CountVectorizer
  corpus = [
       'This is the first document.',
       'This document is the second document.',
       'And this is the third one.',
       'Is this the first document?',
   ]
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(corpus)
  print(X.toarray())
  print(vectorizer.get_feature_names())

The ``CountVectorizer`` class provides a number of different keyword argument options to pre-process text in
various ways which can be passed through ``TMNTVectorizer`` as a dictionary argument ``count_vectorizer_kwargs``::

  corpus = [
       'This is the first document.',
       'This document is the second document.',
       'And this is the third one.',
       'Is this the first document?',
  ]
  vectorizer2 = TMNTVectorizer(count_vectorizer_kwargs={'ngram_range':(2,2)})
  X, _ = vectorizer2.fit_transform(corpus)
  print(X.toarray())
  print(vectorizer2.get_vocab().token_to_idx)

The above snippet uses the ``CountVectorizer`` argument ``ngram_range`` to specify that
bi-grams (pairs of adjacent words) should be used as features rather than single words.

A Slightly More Real Example
++++++++++++++++++++++++++++

The following uses a popular (and now very old) text mining dataset, the 20 Newsgroup dataset,
and demonstrates how to preprocess the data using the ``TMNTVectorizer`` class::

  from sklearn.datasets import fetch_20newsgroups
  from tmnt.preprocess.vectorizer import TMNTVectorizer

Now, download the 20 Newsgroup dataset::

  data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

The document-term matrix ``X`` can be generated using a ``TMNTVectorizer`` object::

  tf_vectorizer = TMNTVectorizer(vocab_size=2000)
  X, _ = tf_vectorizer.fit_transform(data)
  

