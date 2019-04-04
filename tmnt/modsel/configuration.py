# coding: utf-8
import yaml
import io

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

class TMNTConfig(object):

    def __init__(self, c_file):
        self.config_file = c_file
        with io.open(c_file, 'r') as fp:
            self.cd = yaml.safe_load(fp)

    def _get_range_uniform(self, param, cd):
        p = cd[param]
        low = float(p['range'][0])
        upp = float(p['range'][1])
        default_val = p.get('default')
        if default_val:
            default = float(default_val)
        else:
            default = (upp + low) / 2
        use_log = True if ((upp / low) > 100) else False
        return CSH.UniformFloatHyperparameter(param, lower=low, upper=upp, default_value=default, log=use_log)

    def _get_range_integer(self, param, cd, q=1):
        p = cd[param]
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
            default = (upp + low) / 2
        use_log = False
        return CSH.UniformIntegerHyperparameter(param, lower=low, upper=upp, default_value=default, q=q_val, log=use_log)

    def _get_categorical(self, param, cd):
        categories = cd[param]
        return CSH.CategoricalHyperparameter(param, categories)

    def _get_ordinal(self, param, cd):
        values = cd[param]
        return CSH.OrdinalHyperparameter(param, values)

    def get_configspace(self):
        cd = self.cd
        cs = CS.ConfigurationSpace()
        ## learning rate
        lr_c = self._get_range_uniform('lr', cd)
        latent_distribution_c = self._get_categorical('latent_distribution', cd)
        optimizer_c = self._get_categorical('optimizer', cd)
        n_latent_c = self._get_range_integer('n_latent',cd)
        enc_hidden_dim_c = self._get_range_integer('enc_hidden_dim', cd)
        cs.add_hyperparameters([lr_c, latent_distribution_c, optimizer_c, n_latent_c, enc_hidden_dim_c])
        return cs
