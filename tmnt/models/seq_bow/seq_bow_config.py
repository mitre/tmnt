# coding: utf-8
"""
Copyright (c) 2020 The MITRE Corporation.
"""

import autogluon as ag
from tmnt.models.base.base_config import TMNTConfigBase

__all__ = ['TMNTConfigSeqBOW']

class TMNTConfigSeqBOW(TMNTConfigBase):

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
    
