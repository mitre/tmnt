# coding: utf-8
"""
Copyright (c) 2019-2020 The MITRE Corporation.
"""

import yaml
import io
import autogluon as ag

#import ConfigSpace as CS
#import ConfigSpace.hyperparameters as CSH


class TMNTConfig(object):

    def __init__(self, c_file):
        self.config_file = c_file
        with io.open(c_file, 'r') as fp:
            self.cd = yaml.safe_load(fp)

    def _get_range_uniform(self, param, cd):
        if cd.get(param):            
            p = cd[param]
            if len(p['range']) == 1:
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
            if len(p['i_range']) == 1:
                return int(p['i_range'][0])
            low = int(p['i_range'][0])
            upp = int(p['i_range'][1])
            default_val = p.get('default')
            q_val_s = p.get('step')
            if q_val_s:
                q_val = int(q_val_s)
            else:
                q_val = 1
            if default_val:
                default = float(default_val)
            else:
                default = int((upp + low) / 2)
            use_log = False
            if low == upp:
                return low
            else:
                return ag.space.Int(low, upp, default=default)
        else:
            return None

    def _get_categorical(self, param, cd):
        if cd.get(param):
            categories = cd[param]
            return ag.space.Categorical(*categories)
        else:
            return None

    def get_configspace(self):
        cd = self.cd
        sp_dict = {}
        sp_dict['lr'] = self._get_range_uniform('lr', cd)
        sp_dict['latent_distribution'] = self._get_categorical('latent_distribution', cd)
        sp_dict['optimizer'] = self._get_categorical('optimizer', cd)
        sp_dict['n_latent'] = self._get_range_integer('n_latent',cd)
        sp_dict['enc_hidden_dim'] = self._get_range_integer('enc_hidden_dim', cd)
        sp_dict['batch_size'] = self._get_range_integer('batch_size', cd)

        sp_dict['target_sparsity'] = self._get_range_uniform('target_sparsity', cd) or 0.0
        sp_dict['coherence_loss_wt'] = self._get_range_uniform('coherence_loss_wt', cd) or 0.0
        sp_dict['redundancy_loss_wt'] = self._get_range_uniform('redundancy_loss_wt', cd) or 0.0
        sp_dict['num_enc_layers'] = self._get_range_integer('num_enc_layers', cd) or 1
        sp_dict['enc_dr'] = self._get_range_uniform('enc_dr', cd) or 0.0

        embedding_types = cd['embedding_source']
        embedding_space = []
        for et in embedding_types:
            if et == 'random':
                embedding_space.append(ag.space.List('random', self._get_range_integer('embedding_size', cd)))
            else:
                embedding_space.append(ag.space.List(et, self._get_categorical('fixed_embedding', cd)))
        sp_dict['embedding'] = ag.space.Categorical(*embedding_space)

        latent_types = cd['latent_distribution']
        latent_space = []
        for lt in latent_types:
            if lt == 'vmf':
                latent_space.append(ag.space.List('vmf', self._get_range_uniform('kappa', cd)))
            elif lt == 'logistic_gaussian':
                latent_space.append(ag.space.List('logistic_gaussian', self._get_range_uniform('alpha', cd)))
            else:
                latent_space.append(lt)
        sp_dict['latent_dist'] = ag.space.Categorical(*latent_space)

        return sp_dict
