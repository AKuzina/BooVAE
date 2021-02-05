import math
import os
import numpy as np
import torch
import torch.nn as nn
from vae.model.simple_vae import SimpleVAE
from vae.utils.distributions import log_Normal_diag, gaus_kl, bernoulli_kl
from vae.utils.visual_evaluation import plot_images


class Simple(SimpleVAE):
    def __init__(self, z1_size, input_type, ll, arc, **kwargs):
        super(Simple, self).__init__(z1_size, input_type, ll, arc)
        self.num_comp = kwargs['number_components']
        self.incremental = kwargs['incremental']
        self.mog_mu = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.hid_dim))
        self.mog_logvar = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.hid_dim))
        self.init_comp()

        self.learned_mu = nn.ParameterList()
        self.learned_logvar = nn.ParameterList()
        self.component_reconstr_mean = []
        self.component_reconstr_logvar = []

    def log_p_z(self, z):
        z_expand = z.unsqueeze(1)  # MB x 1 x hid
        # mu and logsigma 1 x C х hid

        # MB x C:
        log_comps = log_Normal_diag(z_expand, self.mog_mu, self.mog_logvar, dim=2)
        num_tsk = len(self.learned_mu)
        if self.incremental and num_tsk > 0:
            learned_mu = torch.cat([self.learned_mu[i] for i in range(num_tsk)], 1)
            learned_logs = torch.cat([self.learned_logvar[i] for i in range(num_tsk)], 1)
            log_learned_comps = log_Normal_diag(z_expand, learned_mu, learned_logs, dim=2)
            log_comps = torch.cat((log_comps, log_learned_comps), 1)

        log_comps -= math.log(self.num_comp * (1 + num_tsk))
        log_prior = torch.logsumexp(log_comps, 1)  # MB x 1
        return log_prior

    def generator_regularization(self):
        num_tsk = len(self.learned_mu)
        if num_tsk > 0:
            # learned components = h, (1 x C*num_tsk x hid)
            mu = torch.cat([self.learned_mu[i] for i in range(num_tsk)], dim=1)
            logvar = torch.cat([self.learned_logvar[i] for i in range(num_tsk)], dim=1)
            z_samples = self.reparameterize(mu, logvar)

            N = mu.shape[1]
            # target generator output = x^*, (N x input_size)
            opt_mu = torch.cat(self.component_reconstr_mean, dim=0)

            # current generator output = gen(sample_h)
            curr_mu, curr_logsigma = self.p_x(z_samples)
            curr_mu = curr_mu.reshape(N, -1)
            curr_logsigma = curr_logsigma.reshape(N, -1)

            # current encoder output = enc(x^*), (N x hid)
            curr_z_q_mean, curr_z_q_logvar = self.q_z(opt_mu)

            # generator regularization KL(gen(sample_h) || x^*)
            opt_mu = opt_mu.reshape(N, -1)
            if self.input_type == 'binary':
                reg = bernoulli_kl(curr_mu, opt_mu, dim=1)
            else:
                opt_logsigma = torch.cat(self.component_reconstr_logvar, dim=0).reshape(N, -1)
                reg = gaus_kl(curr_mu, curr_logsigma, opt_mu, opt_logsigma, dim=1)

            # encoder regularization KL(h || enc(x^*))
            reg += gaus_kl(mu.squeeze(0), logvar.squeeze(0), curr_z_q_mean, curr_z_q_logvar, dim=1)
        else:
            reg = torch.zeros(1)
        return reg.mean()/self.hid_dim

    def generate_x(self, N=25):
        num_tsk = len(self.learned_mu)
        mixture_idx = np.random.choice(self.mog_mu.shape[1] * (1 + num_tsk), size=N,
                                       replace=True)

        if self.incremental and num_tsk > 0:
            learned_mu = torch.cat([self.learned_mu[i] for i in range(num_tsk)], 1)
            learned_logs = torch.cat([self.learned_logvar[i] for i in range(num_tsk)], 1)

            total_mu = torch.cat((learned_mu, self.mog_mu), 1)
            total_logvar = torch.cat((learned_logs, self.mog_logvar), 1)
        else:
            total_mu = self.mog_mu
            total_logvar = self.mog_logvar.data
        z_sample_rand = self.reparameterize(total_mu[0, mixture_idx],
                                            total_logvar[0, mixture_idx])

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand

    def add_component(self, from_prev=False):
        mu_to_save = self.mog_mu.clone().detach().data
        logvar_to_save = self.mog_logvar.clone().detach().data
        self.learned_mu.append(nn.Parameter(mu_to_save, requires_grad=False))
        self.learned_logvar.append(nn.Parameter(logvar_to_save, requires_grad=False))
        with torch.no_grad():
            comp_mean, comp_logvar = self.p_x(self.mog_mu)
        self.component_reconstr_mean.append(comp_mean.clone().detach().data.squeeze(0))
        if self.input_type != 'binary':
            self.component_reconstr_logvar.append(comp_logvar.clone().detach().data.squeeze(0))
        self.init_comp()
        print('component added')

    def init_comp(self):
        self.mog_mu.data.normal_(0, 0.5)
        self.mog_logvar.data.fill_(-2)

    def visualize_results(self, args, dir, epoch, real):
        # initialize folders
        self.eval()
        if not os.path.exists(os.path.join(dir, 'reconstruction')):
            os.makedirs(os.path.join(dir, 'reconstruction'))
            plot_images(args, real.data.cpu().numpy(),
                        os.path.join(dir, 'reconstruction'), 'real', size_x=3, size_y=3)

        if not os.path.exists(os.path.join(dir, 'components')):
            os.makedirs(os.path.join(dir, 'components'))

        # plot current reconstructions
        if epoch%20 == 0:
            x_mean = self.reconstruct_x(real)
            plot_images(args, x_mean.data.cpu().numpy(),
                        os.path.join(dir, 'reconstruction'), str(epoch), size_x=3, size_y=3)

            # plot components
            mu = self.mog_mu
            n_tsk = len(self.learned_mu)
            if 'incr' in args.prior and n_tsk > 0:
                old_mu = torch.cat([self.learned_mu[i] for i in range(n_tsk)], 1)
                mu = torch.cat((old_mu, mu), 1)
            out = self.p_x(mu)[0]
            img = out.reshape(mu.shape[1], 1, -1)
            img = img.data.cpu().numpy()[-100:]
            plot_images(args, img, os.path.join(dir, 'components'),
                        str(epoch), size_x=10, size_y=10)
        self.train()

