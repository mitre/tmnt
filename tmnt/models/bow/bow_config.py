# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import autogluon as ag
from tmnt.models.base.base_config import TMNTConfigBase

__all__ = ['TMNTConfigBOW']

class TMNTConfigBOW(TMNTConfigBase):

    def __init__(self, c_file):
        super().__init__(c_file)

    def get_configspace(self):
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
