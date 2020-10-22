# coding: utf-8
"""
Copyright (c) 2019-2020 The MITRE Corporation.
"""

import yaml
import io
import autogluon as ag

__all__ = ['TMNTConfigBase']

class TMNTConfigBase(object):

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
        

