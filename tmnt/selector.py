# coding: utf-8
# Copyright (c) 2020-2021 The MITRE Corporation.
"""
Model selection using AutoGluon.
"""

import pandas as pd
import autogluon.core as ag
import logging
import datetime
import os
import json
import io
import dask
import dask.distributed
from autogluon.core.scheduler.reporter import FakeReporter
from tabulate import tabulate
from tmnt.configuration import TMNTConfigBOW, TMNTConfigSeqBOW
from tmnt.trainer import BowVAETrainer, SeqBowVEDTrainer
from tmnt.utils.log_utils import logging_config
from pathlib import Path

class BaseSelector(object):
    """Base model selector.

    Perform model selection using AutoGluon with any type of topic model.

    Args:
        tmnt_config_space(`tmnt.configuration.TMNTConfigBase`): Object defining the search space for a TMNT topic model
        iterations (int): Number of total model evaluations (default = 12)
        searcher (str): Search algortihm used (random, bayesopt, skopt) (default = random)
        scheduler (str): Scheduler for search (fifo, hyperband) (default = fifo)
        brackets (int): Number of brackets (if using hyperband) (default = 1)
        cpus_per_task (int): Number of cpus to evaluate each model (increase to limit concurrency); (default = 2)
        num_final_evals (int): Number of model refits and evaluations (with different random seeds) using the 
            best found configuration (default = 1)
        rng_seed (int): Random seed used for model selection evaluations (default = 1234)
        log_dir (str): Directory for output of model selection info (default = ./_exps)
    """
    
    def __init__(self, tmnt_config_space, iterations=12, searcher='random', scheduler='fifo', brackets=1, cpus_per_task=2, 
                 num_final_evals=1, rng_seed=1234, log_dir='_exps', max_task_time=120000):
        self.tmnt_config_space = tmnt_config_space
        self.iterations = iterations
        self.searcher = searcher
        self.scheduler = scheduler
        self.brackets = brackets
        self.cpus_per_task = cpus_per_task
        self.num_final_evals = num_final_evals
        self.rng_seed = rng_seed
        self.log_dir = log_dir
        self.max_task_time = max_task_time
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


    def _process_training_history(self, task_dicts, start_timestamp):
        task_dfs = []
        for task_id in task_dicts:
            task_df = pd.DataFrame(task_dicts[task_id])
            task_df = task_df.assign(task_id=task_id,
                                     coherence=task_df.get("coherence", 0.0),
                                     perplexity=task_df.get("perplexity", 0.0),
                                     redundancy=task_df.get("redundancy", 0.0),
                                     runtime=task_df["time_step"] - start_timestamp,
                                     objective=task_df["objective"],
                                     target_epoch=task_df["epoch"].iloc[-1])
            task_dfs.append(task_df)
        if len(task_dfs) > 0:
            result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)
            result = result.sort_values(by="runtime")
            result = result.assign(best=result["objective"].cummax())
        else:
            result = None
        return result
    

    def _select(self, trainer):
        @ag.args(**self.tmnt_config_space)
        def exec_train_fn(args, reporter):
            return trainer.train_model(args, reporter)

        search_options = {
        'num_init_random': 2,
        'debug_log': True}

        num_gpus = 1 if trainer.use_gpu else 0
        if self.scheduler == 'hyperband':
            hpb_scheduler = ag.scheduler.HyperbandScheduler(
                exec_train_fn,
                resource={'num_cpus': self.cpus_per_task, 'num_gpus': num_gpus},
                searcher=self.searcher,
                search_options=search_options,
                num_trials=self.iterations,             #time_out=120,
                time_attr='epoch',
                reward_attr='objective',
                type='stopping',
                grace_period=1,
                reduction_factor=3,
                brackets=self.brackets)
        else:
            hpb_scheduler = ag.scheduler.FIFOScheduler(
                exec_train_fn,
                resource={'num_cpus': self.cpus_per_task, 'num_gpus': num_gpus},
                searcher=self.searcher,
                search_options=search_options,
                num_trials=self.iterations,             #time_out=120
                time_attr='epoch',
                reward_attr='objective',
                )
        hpb_scheduler.run()
        hpb_scheduler.join_jobs()
        return hpb_scheduler


    def select_model(self, trainer):
        """Select best model using the given model trainer.

        Performs model selection using AutoGluon according to the provided configuration space
        and various model selection options, e.g. searcher and scheduler, number of iterations

        Args:
            trainer (:class:`tmnt.trainer.BaseTrainer`): A trainer that fits and evaluates models given a configuration.
        Returns:
            estimator (:class:`tmnt.estimator.BaseEstimator`: A fit estimator for the final selected model hyper-parameters
        """
        dd = datetime.datetime.now()
        scheduler = self._select(trainer)
        best_config_spec = scheduler.get_best_config()
        args_dict = ag.space.Dict(**scheduler.train_fn.args)
        best_config = args_dict.sample(**best_config_spec)
        logging.info("Best configuration = {}".format(best_config))
        logging.info("Best configuration objective = {}".format(scheduler.get_best_reward()))
        best_config_dict = ag.space.Dict(**best_config)
        out_config_file  = os.path.join(self.log_dir, 'best.model.config')
        with open(out_config_file, 'w') as fp:
            specs = json.dumps(best_config)
            fp.write(specs)
        logging.info("===> Writing best configuration to {}".format(out_config_file))            
        logging.info("******************************* RETRAINING WITH BEST CONFIGURATION **************************")
        estimator, obj, vres, _ = trainer.train_with_single_config(best_config_dict, self.num_final_evals)
        logging.info("Objective with final retrained model/estimator: {}".format(obj))
        logging.info("Writing model to: {}".format(trainer.model_out_dir))
        trainer.write_model(estimator)        
        dd_finish = datetime.datetime.now()
        logging.info("Model selection run FINISHED. Time: {}".format(dd_finish - dd))
        results_df = self._process_training_history(scheduler.training_history.copy(),
                                                    start_timestamp=scheduler._start_time)
        if results_df is not None:
            logging.info("Printing hyperparameter results")
            out_html = os.path.join(self.log_dir, 'selection.html')
            results_df.to_html(out_html)
            out_pretty = os.path.join(self.log_dir, 'selection.table.txt')
            with io.open(out_pretty, 'w') as fp:
                fp.write(tabulate(results_df, headers='keys', tablefmt='pqsl'))
        else:
            logging.info("Training history unavailable")
        return estimator, obj, vres


def model_select_bow_vae(c_args):
    logging_config(folder=c_args.save_dir, name='tmnt', level=c_args.log_level, console_level=c_args.log_level)
    ## dask config overrides
    dask.config.config['distributed']['worker']['use-file-locking'] = False
    dask.config.config['distributed']['comm']['timeouts']['connect'] = '90s'
    ##
    tmnt_config = TMNTConfigBOW(c_args.config_space).get_configspace()
    trainer = BowVAETrainer.from_arguments(c_args, val_each_epoch = (not (c_args.searcher == 'random')))
    selector = BaseSelector(tmnt_config,
                            iterations      = c_args.iterations,
                            searcher        = c_args.searcher,
                            scheduler       = c_args.scheduler,
                            brackets        = c_args.brackets,
                            cpus_per_task   = c_args.cpus_per_task,
                            num_final_evals = c_args.num_final_evals,
                            rng_seed        = c_args.seed,
                            log_dir         = trainer.log_out_dir)
    sources = [ e['source'] for e in tmnt_config.get('embedding').data if e['source'] != 'random' ]
    logging.info('>> Pre-caching pre-trained embeddings/vocabularies: {}'.format(sources))
    trainer.pre_cache_vocabularies(sources)
    selector.select_model(trainer)
        

def model_select_seq_bow(c_args):
    logging_config(folder=c_args.save_dir, name='tmnt', level=c_args.log_level, console_level=c_args.log_level)
    tmnt_config = TMNTConfigSeqBOW(c_args.config_space).get_configspace()
    trainer = SeqBowVEDTrainer.from_arguments(c_args, tmnt_config)
    selector = BaseSelector(tmnt_config,
                            iterations=c_args.iterations,
                            searcher=c_args.searcher,
                            scheduler=c_args.scheduler,
                            brackets=c_args.brackets,
                            cpus_per_task=c_args.cpus_per_task,
                            num_final_evals=c_args.num_final_evals,
                            rng_seed=c_args.seed,
                            log_dir=trainer.model_out_dir)
    selector.select_model(trainer)
    
        
