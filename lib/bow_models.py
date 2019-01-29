# coding: utf-8

from mxnet.gluon import HybridBlock

class BowNTM(HybridBlock):

    def __init__(self, batch_size, vocab_size, l1_dim, n_latent, gen_layers=4, prefix=prefix, params=params):
        super(BowNTM, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential()
            self.encoder.add(gluon.nn.Dense(units = l1_dim, activation='tanh'))
            self.encoder.add(gluon.nn.Dense(units = n_latent*2, activation='tanh'))
            self.generator = gluon.nn.HybridSequential()
            with self.generator.name_scope():
                for i in range(gen_layers):
                    self.generator.add(gluon.nn.Dense(units=n_latent, activation='tanh'))
            self.decoder = gluon.nn.Dense(in_units=n_latent, units=vocab_size)


    def hybrid_forward(self, F, data):
        ## data should have shame N x V
        enc_out = self.encoder(data)
        mu_lv = F.split(enc_out, axis=1, num_outputs=2) ## split in half along final dimension
        mu = mu_lv[0]
        lv = mu_lv[1]
        ## Standard Gaussian VAE/VED
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=self.model_ctx)
        z = mu + F.exp(0.5*lv)*eps
        y = self.decoder(z)
        KL = 0.5*F.sum(1+lv-mu*mu-F.exp(lv),axis=1)

        gen_out = self.generator(z)  ## just mu when predicting??
        res = gen_out + z

        dec_out = decoder(res)
        ## TODO:
        ## get reconstruction loss
        ## get L1 penalty
