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
        if cd.get(param):            
            p = cd[param]
            if len(p['range']) == 1:
                return CSH.Constant(param, float(p['range'][0]))
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
            return CSH.UniformFloatHyperparameter(param, lower=low, upper=upp, default_value=default, log=use_log)
        else:
            return None

    def _get_range_integer(self, param, cd, q=1):
        p = cd[param]
        
        if len(p['i_range']) < 2:
            return CSH.Constant(param, int(p['i_range'][0]))
        
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
            return CSH.Constant(param, low)
        else:
            return CSH.UniformIntegerHyperparameter(param, lower=low, upper=upp, default_value=default, q=q_val, log=use_log)

    def _get_categorical(self, param, cd):
        if cd.get(param):
            categories = cd[param]
            return CSH.CategoricalHyperparameter(param, categories)
        else:
            return None

    def _get_ordinal(self, param, cd):
        values = cd[param]
        return CSH.OrdinalHyperparameter(param, values)

    def get_configspace(self):
        cd = self.cd
        cs = CS.ConfigurationSpace()
        lr_c = self._get_range_uniform('lr', cd)
        latent_distribution_c = self._get_categorical('latent_distribution', cd)
        optimizer_c = self._get_categorical('optimizer', cd)
        n_latent_c = self._get_range_integer('n_latent',cd)
        enc_hidden_dim_c = self._get_range_integer('enc_hidden_dim', cd)
        
        kappa_c = self._get_range_uniform('kappa', cd)

        batch_size_c = self._get_range_integer('batch_size', cd)

        cs.add_hyperparameters([batch_size_c, lr_c, latent_distribution_c, optimizer_c, n_latent_c, enc_hidden_dim_c, kappa_c])

        ## optional hyperparameters
        target_sparsity_c = self._get_range_uniform('target_sparsity', cd)
        if target_sparsity_c:
            cs.add_hyperparameters([target_sparsity_c])

        coherence_reg_penalty_c = self._get_range_uniform('coherence_regularizer_penalty', cd)
        if coherence_reg_penalty_c:
            cs.add_hyperparameters([coherence_reg_penalty_c])

        embedding_source_c = self._get_categorical('embedding_source', cd)
        if embedding_source_c:
            cs.add_hyperparameters([embedding_source_c])
            fixed_embedding_c = self._get_categorical('fixed_embedding', cd)
            if fixed_embedding_c:
                cs.add_hyperparameters([fixed_embedding_c])
            if embedding_source_c.is_legal('random'):  ## if NOT, then don't add embedding size as a hyperparameters at all as it's determined by the embedding
                embedding_size_c = self._get_range_integer('embedding_size', cd)
                cs.add_hyperparameters([embedding_size_c])
                cond_embed = CS.EqualsCondition(embedding_size_c, embedding_source_c, 'random')
                cs.add_condition(cond_embed)
        else:
            ## add embedding size if no source is specified
            embedding_size_c = self._get_range_integer('embedding_size', cd)
            cs.add_hyperparameters([embedding_size_c])

        cond_kappa = CS.EqualsCondition(kappa_c, latent_distribution_c, 'vmf')
        cs.add_condition(cond_kappa) # only use kappa_c if condition is met
        return cs
