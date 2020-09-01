# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

class BaseVAE(object):

    def get_topic_vectors(self):
        raise NotImplementedError()


    def _get_model(self):
        """
        Returns
        -------
        MXNet model initialized using provided hyperparameters
        """

        raise NotImplementedError()


    def fit(self, X, y):
        """
        Fit VAE model according to the given training data X with optional co-variates y.
  
        Parameters
        ----------
        X: tensor representing input data

        y: tensor representing covariate/labels associated with data elements
        """

        raise NotImplementedError()

    def fit_with_validation(self, X, y, val_X, val_Y):
        """
        Fit VAE model according to the given training data X with optional co-variates y;
        validate (potentially each epoch) with validation data val_X and optional co-variates val_Y
  
        Parameters
        ----------
        X: tensor representing training data

        y: tensor representing covariate/labels associated with data elements in training data

        val_X: tensor representing validation data

        val_y: tensor representing covariate/labels associated with data elements in validation data
        """

        raise NotImplementedError()
