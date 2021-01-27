# coding: utf-8
# Copyright (c) 2019-2020 The MITRE Corporation.
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
            categories = cd[param]
            return ag.space.Categorical(*categories)
        else:
            return None

    def get_configspace(self):
        """Returns a dictionary with a point/sample in the configspace.
        Returns:
            (dict): Dictionary with configuration parameter values.
        """
        raise NotImplemented


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


class TMNTConfigSeqBOW(BaseTMNTConfig):

    def __init__(self, c_file):
        super().__init__(c_file)

    def get_configspace(self):
        cd = self.cd
        sp_dict = {}
        sp_dict['epochs'] = int(cd['epochs'])
        sp_dict['gen_lr'] = self._get_range_uniform('gen_lr', cd)
        sp_dict['min_lr'] = self._get_range_uniform('min_lr', cd)
        sp_dict['dec_lr'] = self._get_range_uniform('dec_lr', cd)
        sp_dict['n_latent'] = self._get_range_integer('n_latent', cd)
        sp_dict['batch_size'] = self._get_range_integer('batch_size', cd)
        sp_dict['optimizer'] = self._get_categorical('optimizer', cd)
        sp_dict['warmup_ratio'] = self._get_range_uniform('warmup_ratio', cd)
        sp_dict['embedding_source'] = self._get_categorical('embedding_source', cd)
        sp_dict['redundancy_reg_penalty'] = self._get_range_uniform('redundancy_reg_penalty', cd)
        sp_dict['sent_size'] = int(cd['sent_size'])  ## currently not considering different sent_size values as this requires re-prepping the data
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
    
        
