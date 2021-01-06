# coding: utf-8
# Copyright (c) 2020 The MITRE Corporation.
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
from autogluon.core.scheduler.reporter import FakeReporter
from tabulate import tabulate
from tmnt.configuration import TMNTConfigBOW, TMNTConfigSeqBOW
from tmnt.trainer import BowVAETrainer, SeqBowVEDTrainer

class BaseSelector(object):
    """Base model selector.

    Perform model selection using AutoGluon with any type of topic model.

    Args:
        tmnt_config_space(`tmnt.configuration.TMNTConfigBase`): Object defining the search space for a TMNT topic model
        iterations (int): Number of total model evaluations
        searcher (str): Search algortihm used (random, bayesopt, skopt)
        scheduler (str): Scheduler for search (fifo, hyperband)
        brackets (int): Number of brackets (if using hyperband)
        cpus_per_task (int): Number of cpus to evaluate each model (increase to limit concurrency)
        use_gpu (bool): Whether to use GPUs when training each model
        num_final_evals (int): Number of model refits and evaluations (with different random seeds) using the best found configuration
        rng_seed (int): Random seed used for model selection evaluations
        log_dir (str): Directory for output of model selection info
    """
    
    def __init__(self, tmnt_config_space, iterations, searcher, scheduler, brackets, cpus_per_task, use_gpu,
                 num_final_evals, rng_seed, log_dir):
        self.log_dir = log_dir
        self.tmnt_config_space = tmnt_config_space
        self.iterations = iterations
        self.searcher = searcher
        self.scheduler = scheduler
        self.brackets = brackets
        self.cpus_per_task = cpus_per_task
        self.use_gpu = use_gpu
        self.num_final_evals = num_final_evals
        self.rng_seed = rng_seed


    def _process_training_history(self, task_dicts, start_timestamp):
        task_dfs = []
        for task_id in task_dicts:
            task_df = pd.DataFrame(task_dicts[task_id])
            task_df = task_df.assign(task_id=task_id,
                                     coherence=task_df["coherence"],
                                     perplexity=task_df["perplexity"],
                                     redundancy=task_df["redundancy"],
                                     runtime=task_df["time_step"] - start_timestamp,
                                     objective=task_df["objective"],
                                     target_epoch=task_df["epoch"].iloc[-1])
            task_dfs.append(task_df)
        result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)
        result = result.sort_values(by="runtime")
        result = result.assign(best=result["objective"].cummax())
        return result
    

    def _select(self, trainer):
        @ag.args(**self.tmnt_config_space)
        def exec_train_fn(args, reporter):
            return trainer.train_model(args, reporter)

        search_options = {
        'num_init_random': 2,
        'debug_log': True}

        num_gpus = 1 if self.use_gpu else 0
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
        """
        dd = datetime.datetime.now()
        scheduler = self._select(trainer)
        best_config_spec = scheduler.get_best_config()
        args_dict = ag.space.Dict(**scheduler.train_fn.args)
        best_config = args_dict.sample(**best_config_spec)
        logging.info("Best configuration = {}".format(best_config))
        logging.info("Best configuration objective = {}".format(scheduler.get_best_reward()))
        best_config_dict = ag.space.Dict(**best_config)
        logging.info("******************************* RETRAINING WITH BEST CONFIGURATION **************************")
        model, obj = trainer.train_with_single_config(best_config_dict, self.num_final_evals)
        logging.info("Objective with final retrained model: {}".format(obj))
        trainer.write_model(model, best_config)
        with open(os.path.join(self.log_dir, 'best.model.config'), 'w') as fp:
            specs = json.dumps(best_config)
            fp.write(specs)
        dd_finish = datetime.datetime.now()
        logging.info("Model selection run FINISHED. Time: {}".format(dd_finish - dd))
        results_df = self._process_training_history(scheduler.training_history.copy(),
                                                    start_timestamp=scheduler._start_time)
        logging.info("Printing hyperparameter results")
        out_html = os.path.join(self.log_dir, 'selection.html')
        results_df.to_html(out_html)
        out_pretty = os.path.join(self.log_dir, 'selection.table.txt')
        with io.open(out_pretty, 'w') as fp:
            fp.write(tabulate(results_df, headers='keys', tablefmt='pqsl'))
        ## scheduler.shutdown()


def model_select_bow_vae(c_args):
    tmnt_config = TMNTConfigBOW(c_args.config_space).get_configspace()
    trainer = BowVAETrainer.from_arguments(c_args, val_each_epoch = (not (c_args.searcher == 'random')))
    selector = BaseSelector(tmnt_config,
                            c_args.iterations,
                            c_args.searcher,
                            c_args.scheduler,
                            c_args.brackets,
                            c_args.cpus_per_task,
                            c_args.use_gpu,
                            c_args.num_final_evals,
                            c_args.seed,
                            trainer.log_out_dir)
    sources = [ e['source'] for e in tmnt_config.get('embedding').data if e['source'] != 'random' ]
    logging.info('>> Pre-caching pre-trained embeddings/vocabularies: {}'.format(sources))
    trainer.pre_cache_vocabularies(sources)
    selector.select_model(trainer)
        

def model_select_seq_bow(c_args):
    tmnt_config = TMNTConfigSeqBOW(c_args.config_space).get_configspace()
    trainer = SeqBowVEDTrainer.from_arguments(c_args)
    selector = BaseSelector(tmnt_config,
                            c_args.iterations,
                            c_args.searcher,
                            c_args.scheduler,
                            c_args.brackets,
                            c_args.cpus_per_task,
                            c_args.use_gpu,
                            c_args.num_final_evals,
                            c_args.seed,
                            trainer.model_out_dir)
    selector.select_model(trainer)
    
        
