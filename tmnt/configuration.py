# coding: utf-8
# Copyright (c) 2019-2021 The MITRE Corporation.
"""
Configuration objects for representing hyperparameters faciliating model selection.
"""

import yaml
import io
import autogluon.core as ag

__all__ = ['BaseTMNTConfig']

class BaseTMNTConfig(object):
    """Base configuration (space) for TMNT
    
    Parameters:
        c_file (str): String path to configuration space (.yaml) file
    """

    def __init__(self, c_file):
        self.config_file = c_file
        with io.open(c_file, 'r') as fp:
            self.cd = yaml.safe_load(fp)

    def _get_range_uniform(self, param, cd):
        if cd.get(param):            
            p = cd[param]
            if isinstance(p, float):
                return p
            elif isinstance(p, int):
                return float(p)
            elif len(p['range']) == 1:
                return float(p['range'][0])
            low = float(p['range'][0])
            upp = float(p['range'][1])
            default_val = p.get('default')
            if default_val:
                default = float(default_val)
            else:
                default = (upp + low) / 2
            use_log = False
            if ( (low != 0.0) and (abs(upp / low) >= 1000) ):
                use_log = True
            return ag.space.Real(low, upp, default=default, log=use_log)
        else:
            return None

    def _get_range_integer(self, param, cd, q=1):
        if cd.get(param):
            p = cd[param]
            if isinstance(p, int):
                return p
            if len(p['i_range']) == 1:
                return int(p['i_range'][0])
            low = int(p['i_range'][0])
            upp = int(p['i_range'][1])
            default_val = p.get('default')
            q_val_s = p.get('step')
            if default_val:
                default = float(default_val)
            else:
                default = int((upp + low) / 2)
            if q_val_s:
                ivals = list(range(low, upp+1, int(q_val_s)))
                return ag.space.Categorical(*ivals)
            else:
                q_val = 1                
                if low == upp:
                    return low
                else:
                    return ag.space.Int(low, upp, default=default)
        else:
            return None

    def _get_categorical(self, param, cd):
        if cd.get(param):
            v = cd[param]
            if isinstance(v, str):
                return v
            return ag.space.Categorical(*v)
        else:
            return None

    def _get_atomic(self, param, cd):
        if cd.get(param):
            return cd[param]
        else:
            return None

    def get_configspace(self):
        """Returns a dictionary with a point/sample in the configspace.
        Returns:
            (dict): Dictionary with configuration parameter values.
        """
        raise NotImplemented

default_bow_config_space = {
    'epochs': 27,
    'gamma': 1.0,
    'multilabel': False,
    'lr': ag.space.Real(1e-4, 1e-2),
    'batch_size': ag.space.Categorical(*list(range(100,401,100))),
    'latent_distribution': ag.space.Categorical(ag.space.Dict(**{'dist_type': 'vmf', 'kappa': ag.space.Real(1.0, 100.0)}),
                                                ag.space.Dict(**{'dist_type': 'logistic_gaussian', 'alpha': ag.space.Real(0.5, 5.0)})),
    'optimizer': ag.space.Categorical('adam'),
    'n_latent': ag.space.Categorical(15,20,25),
    'enc_hidden_dim': ag.space.Categorical(50,100,150,200),
    'embedding': ag.space.Categorical(ag.space.Dict(**{'source': 'random', 'size': ag.space.Categorical(50,100,150,200)})),
    'coherence_loss_wt': 0.0,
    'redundancy_loss_wt': 0.0,
    'num_enc_layers': 1,
    'enc_dr': ag.space.Real(0.0, 0.2),
    'covar_net_layers': 1
    }

class TMNTConfigBOW(BaseTMNTConfig):

    def __init__(self, c_file):
        super().__init__(c_file)

    def get_configspace(self):
        """Get a dictionary representing this config space based on the config file provided
        during object creation.
        """
        cd = self.cd
        sp_dict = {}
        sp_dict['epochs'] = int(cd['epochs'])
        sp_dict['gamma'] = self._get_range_uniform('gamma', cd)
        sp_dict['multilabel'] = self._get_atomic('multilabel', cd)
        sp_dict['lr'] = self._get_range_uniform('lr', cd)
        sp_dict['optimizer'] = self._get_categorical('optimizer', cd)
        sp_dict['n_latent'] = self._get_range_integer('n_latent',cd)
        sp_dict['enc_hidden_dim'] = self._get_range_integer('enc_hidden_dim', cd)
        sp_dict['batch_size'] = self._get_range_integer('batch_size', cd)
        sp_dict['coherence_loss_wt'] = self._get_range_uniform('coherence_loss_wt', cd) or 0.0
        sp_dict['redundancy_loss_wt'] = self._get_range_uniform('redundancy_loss_wt', cd) or 0.0
        sp_dict['num_enc_layers'] = self._get_range_integer('num_enc_layers', cd) or 1
        sp_dict['enc_dr'] = self._get_range_uniform('enc_dr', cd) or 0.0
        sp_dict['covar_net_layers'] = self._get_range_integer('covar_net_layers', cd) or 1
        sp_dict['classifier_dropout'] = self._get_range_uniform('classifier_dropout', cd) or 0.1

        embedding_types = cd['embedding']
        embedding_space = []        
        for et in embedding_types:
            if et['source'] == 'random':
                embedding_space.append(ag.space.Dict(**{'source': 'random', 'size': self._get_range_integer('size', et)}))
            else:
                fixed_assigned = et.get('fixed')
                if fixed_assigned is None:
                    embedding_space.append(ag.space.Dict(**{'source': et['source'], 'fixed': ag.space.Bool()}))
                else:
                    embedding_space.append(ag.space.Dict(**{'source': et['source'], 'fixed': fixed_assigned.lower()}))
        sp_dict['embedding'] = ag.space.Categorical(*embedding_space)

        latent_types = cd['latent_distribution']
        latent_space = []
        for lt in latent_types:
            dist_type = lt['dist_type']
            if dist_type == 'vmf':
                latent_space.append(ag.space.Dict(**{'dist_type': 'vmf', 'kappa': self._get_range_uniform('kappa', lt)}))
            elif dist_type == 'logistic_gaussian':
                latent_space.append(ag.space.Dict(**{'dist_type': 'logistic_gaussian', 'alpha': self._get_range_uniform('alpha', lt)}))
            else:
                latent_space.append(ag.space.Dict(**{'dist_type': 'gaussian'}))
        sp_dict['latent_distribution'] = ag.space.Categorical(*latent_space)
        return sp_dict


class TMNTConfigBOWMetric(TMNTConfigBOW):
    def __init__(self, c_file):
        super().__init__(c_file)

    def get_configspace(self):
        cd = self.cd
        sp_dict = super().get_configspace()
        sp_dict['sdml_smoothing_factor'] = self._get_range_uniform('sdml_smoothing_factor', cd)
        return sp_dict
    


default_seq_config_space = {
    'epochs': 3,
    'gamma': 1.0,
    'multilabel': False,
    'bert_model_name': 'bert_12_768_12',
    'bert_dataset': 'book_corpus_wiki_en_uncased',
    'lr': ag.space.Real(1e-5, 1e-4),
    'decoder_lr': ag.space.Real(0.00005, 0.002),
    'latent_distribution': ag.space.Categorical(ag.space.Dict(**{'dist_type': 'vmf', 'kappa': ag.space.Real(20.0, 80.0)}),
                                                ag.space.Dict(**{'dist_type': 'logistic_gaussian', 'alpha': ag.space.Real(0.5, 5.0)})),
    'optimizer': ag.space.Categorical('bertadam'),
    'n_latent': ag.space.Categorical(15,20,25),
    'max_seq_len': 128,
    'warmup_ratio': ag.space.Real(0.05, 0.2),
    'classifier_dropout': 0.2,
    'batch_size': 8
    }
    

class TMNTConfigSeqBOW(BaseTMNTConfig):

    def __init__(self, c_file):
        super().__init__(c_file)

    def get_configspace(self):
        cd = self.cd
        sp_dict = {}
        sp_dict['epochs'] = int(cd['epochs'])
        sp_dict['gamma']  = self._get_range_uniform('gamma', cd)
        sp_dict['lr'] = self._get_range_uniform('lr', cd)
        sp_dict['min_lr'] = self._get_range_uniform('min_lr', cd)
        sp_dict['decoder_lr'] = self._get_range_uniform('decoder_lr', cd)
        sp_dict['n_latent'] = self._get_range_integer('n_latent', cd)
        sp_dict['batch_size'] = self._get_range_integer('batch_size', cd)
        sp_dict['optimizer'] = self._get_categorical('optimizer', cd)
        sp_dict['warmup_ratio'] = self._get_range_uniform('warmup_ratio', cd)
        sp_dict['embedding_source'] = self._get_categorical('embedding_source', cd)
        sp_dict['redundancy_reg_penalty'] = self._get_range_uniform('redundancy_reg_penalty', cd)
        sp_dict['max_seq_len'] = self._get_range_integer('max_seq_len', cd)
        sp_dict['bert_model_name'] = self._get_categorical('bert_model_name', cd)
        sp_dict['bert_dataset'] = self._get_categorical('bert_dataset', cd)
        sp_dict['use_labels'] = self._get_atomic('use_labels', cd)
        sp_dict['classifier_dropout'] = self._get_range_uniform('classifier_dropout', cd)
        latent_types = cd['latent_distribution']
        latent_space = []
        for lt in latent_types:
            dist_type = lt['dist_type']
            if dist_type == 'vmf':
                latent_space.append(ag.space.Dict(**{'dist_type': 'vmf', 'kappa': self._get_range_uniform('kappa', lt)}))
            elif dist_type == 'logistic_gaussian':
                latent_space.append(ag.space.Dict(**{'dist_type': 'logistic_gaussian', 'alpha': self._get_range_uniform('alpha', lt)}))
            else:
                latent_space.append(ag.space.Dict(**{'dist_type': 'gaussian'}))
        sp_dict['latent_distribution'] = ag.space.Categorical(*latent_space)
        return sp_dict


class TMNTConfigSeqBOWMetric(TMNTConfigSeqBOW):

    def __init__(self, c_file):
        super().__init__(c_file)

    def get_configspace(self):
        cd = self.cd
        sp_dict = super().get_configspace()
        sp_dict['sdml_smoothing_factor'] = self._get_range_uniform('sdml_smoothing_factor', cd)
        return sp_dict
    
        
