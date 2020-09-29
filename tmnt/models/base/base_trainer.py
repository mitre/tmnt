# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import logging
import mxnet as mx
import datetime
import time
from tmnt.utils.random import seed_rng
from autogluon.scheduler.reporter import FakeReporter

class BaseTrainer(object):


    def __init__(self, train_data, test_data, train_labels, test_labels, rng_seed):
        self.train_data   = train_data
        self.test_data    = test_data
        self.train_labels = train_labels
        self.test_labels  = test_labels
        self.rng_seed     = rng_seed


    def x_get_mxnet_visible_gpus(self):

        gpu_count = 0
        while True:
            try:
                arr = mx.np.array(1.0, ctx=mx.gpu(gpu_count))
                arr.asnumpy()
                gpu_count += 1
            except Exception:
                break
        return [mx.gpu(i) for i in range(gpu_count)]

    def _get_mxnet_visible_gpus(self):
        ln = 0
        t = datetime.datetime.now()
        while ln < 1 and ((datetime.datetime.now() - t).seconds < 30):
            time.sleep(1)
            gpus = self.x_get_mxnet_visible_gpus()
            ln = len(gpus)
        if ln > 0:
            return gpus
        else:
            raise Exception("Unable to get a gpu after 30 tries")
        

    def train_with_single_config(self, best_config, num_evals):
        rng_seed = self.rng_seed
        best_obj = -1000000000.0
        best_model = None
        if self.test_data is not None:
            #if c_args.tst_vec_file:
            #    trainer.set_heldout_data_as_test()        
            logging.info("Training with config: {}".format(best_config))
            npmis, perplexities, redundancies, objectives = [],[],[],[]
            ntimes = int(num_evals)
            for i in range(ntimes):
                seed_rng(rng_seed) # update RNG
                rng_seed += 1
                model, obj, npmi, perplexity, redundancy = self.train_model(best_config, FakeReporter())
                npmis.append(npmi)
                perplexities.append(perplexity)
                redundancies.append(redundancy)
                objectives.append(obj)
                if obj > best_obj:
                    best_obj = obj
                    best_model = model
            #test_type = "HELDOUT" if c_args.tst_vec_file else "VALIDATION"
            test_type = "VALIDATION"
            if ntimes > 1:
                logging.info("Final {} NPMI         ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(npmis), statistics.stdev(npmis)))
                logging.info("Final {} Perplexity   ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(perplexities), statistics.stdev(perplexities)))
                logging.info("Final {} Redundancy   ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(redundancies), statistics.stdev(redundancies)))
                logging.info("Final {} Objective    ==> Mean: {}, StdDev: {}"
                             .format(test_type, statistics.mean(objectives), statistics.stdev(objectives)))
            else:
                logging.info("Final {} NPMI         ==> {}".format(test_type, npmis[0]))
                logging.info("Final {} Perplexity   ==> {}".format(test_type, perplexities[0]))
                logging.info("Final {} Redundancy   ==> {}".format(test_type, redundancies[0]))
                logging.info("Final {} Objective    ==> {}".format(test_type, objectives[0]))            
            return best_model, best_obj
        else:
            model, obj, _, _, _ = self.train_model(best_config, FakeReporter())
            return model, obj

        

    def train_model(self, config, reporter):
        """
        Parameters
        ----------
        
        """
        raise NotImplementedError()
