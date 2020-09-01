# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

class BaseSelector(object):

    def select_model(dataset, config_space):
        raise NotImplementedError()
