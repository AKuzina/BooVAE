import math
import os
import numpy as np
import torch
import torch.nn as nn
from vae.model.simple_vae import SimpleVAE
from vae.utils.distributions import log_Normal_diag, gaus_kl, bernoulli_kl
from vae.utils.nn import NonLinear
from vae.utils.mixture import VampMixture
from vae.utils.visual_evaluation import plot_images


class Simple(SimpleVAE):
    def __init__(self, z1_size, input_type, ll, arc,  **kwargs):
        super(Simple, self).__init__(z1_size, input_type, ll, arc)
        self.scale = kwargs['scale']
        self.comp_weight = kwargs['comp_weight']
        self.pseudoinputs_mean = kwargs['pseudoinputs_mean']
        self.pseudoinputs_std = kwargs['pseudoinputs_std']
        self.X_opt = kwargs['X_opt']
        self.input_size = kwargs['input_size']
        self.comp_size = kwargs['input_size'].copy()
        if self.input_type == 'continuous':
            self.comp_size[1] //= self.scale
            self.comp_size[2] //= self.scale
        if 'min_inp' not in kwargs.keys():
            min_inp = self.X_opt.min().item()
        else:
            min_inp = kwargs['min_inp']
        max_inp = self.X_opt.max().item()
        # now they are in the space of inputs (like in vamp)
        act = nn.Hardtanh(min_val=min_inp, max_val=max_inp)

        if self.scale > 1:
            self.pseudo_prep = nn.Sequential(nn.Upsample(scale_factor=self.scale),
                                             act)
        else:
            self.pseudo_prep = nn.Sequential(act)

        self.h_mu_f = None
        mean_opt = self.X_opt.mean(0, keepdim=True)
        self.h_mu = nn.Parameter(torch.FloatTensor(torch.Size([1]) + torch.Size(self.comp_size)))
        self.prior = VampMixture(pseudoinputs=[mean_opt], alpha=[1.])

    def init_prior(self):
        # if self.h_mu_f.requires_grad:
        if self.h_mu_f is not None:
            mu = nn.Parameter(self.h_mu_f(self.idle_input), requires_grad=False)
            mu = mu.reshape([-1] + self.input_size)
            self.prior.add_component(mu, alpha=1)
            self.h_mu_f = None
        self.reset_parameters()

    def log_gaus_mixture(self, sample, means, logvars, weights):
        """
        Compute log density of gaussian mixture
        :param sample: MB x dim
        :param means: C x dim
        :param logvars: C x dim
        :param weights: C
        :return: log( \sum_i w_i p_i(sample) )
        """
        sample = sample.unsqueeze(1)  # MB x 1 x dim
        means = means.unsqueeze(0)  # 1 x C x dim
        logvars = logvars.unsqueeze(0)  # 1 x C x dim

        log_w = torch.log(weights).unsqueeze(0).to(sample.device)  # 1 x C
        log_comps = log_Normal_diag(sample, means, logvars, dim=2)  # MB x C
        log_density = torch.logsumexp(log_comps + log_w, dim=1)  # MB x 1
        return log_density

    def get_current_prior(self, curr_task=False):
        """
        Project all the component into latent space
        :return: components and their weights
        """
        init_tr = self.training
        if init_tr:
            self.eval()
        c = len(self.prior.mu_list)  # number of learned components
        n_prev = len(self.prior.pr_q_means)

        if curr_task:
            n_comp = c - n_prev
        else:
            n_comp = c
        if c > 0:
            psinp = self.prior.mu_list[-n_comp:]
            if self.h_mu_f is not None:
                psinp.append(self.h_mu_f(self.idle_input))
            X = torch.cat(psinp, 0)  # C x inp
            X = X.reshape([-1] + self.input_size)
            means, logvars = self.q_z(X)  # C x hid
            w = self.prior.weights[-n_comp:].clone()
            w /= w.sum()
        else:
            inp = self.h_mu_f(self.idle_input).reshape([-1] + self.input_size)
            means, logvars = self.q_z(inp)
            w = torch.Tensor([1])
        if init_tr:
            self.train()
        return means, logvars, w

    def log_p_z(self, z_sample):
        # z: MB x hid
        self.prior.mu_list[0] = self.prior.mu_list[0].to(z_sample.device)
        self.X_opt = self.X_opt.to(z_sample.device)
        if self.training:
            means, logvars, w = self.get_current_prior(curr_task=True)
        else:
            means, logvars, w = self.get_current_prior(curr_task=False)
        log_prior = self.log_gaus_mixture(z_sample, means, logvars, w)
        return log_prior

    def opt_prior(self, z_sample, pr_means=None, pr_logvars=None, pr_w=None):

        c = self.mean_opt.shape[0]
        w = torch.ones(c)/c
        log_opt_pr = self.log_gaus_mixture(z_sample, self.mean_opt, self.logvar_opt, w)

        if self.prior.num_tasks > 1 and pr_means is not None:
            n_tasks = self.prior.num_tasks
            prev_tasks = (self.prior.task_weight != n_tasks).nonzero().squeeze(1)

            means = pr_means[prev_tasks]
            logvars = pr_logvars[prev_tasks]
            w = pr_w[prev_tasks].clone()
            w /= w.sum()

            log_prior = self.log_gaus_mixture(z_sample, means, logvars, w)

            # sum 2 densities
            log_opt_pr = torch.logsumexp(torch.stack([log_prior + math.log(n_tasks - 1),
                                                      log_opt_pr], 1), 1)
            log_opt_pr = log_opt_pr - math.log(n_tasks)
        return log_opt_pr

    def generator_regularization(self):
        init_tr = self.training
        if init_tr:
            self.eval()
        N = len(self.prior.reconstruction_means)
        if N > 0:
            weights = self.prior.weights[:N] / (self.prior.num_tasks - 1)

            # decoder reg
            #  1. Correct components inlatent space
            z_mu_corr, z_logvar_corr = torch.cat(self.prior.pr_q_means), torch.cat(
                self.prior.pr_q_logvars)
            sample_correct = self.reparameterize(z_mu_corr,
                                                 torch.clamp(z_logvar_corr, -5, -2))
            # sample_correct = z_mu_corr
            x_mu, x_logvar = self.p_x(sample_correct)
            MB = x_mu.shape[0]

            #  2. Correct reconstruction
            x_mu_corr, x_logvar_corr = torch.cat(self.prior.reconstruction_means),  \
                                       torch.cat(self.prior.reconstruction_logvars)
            if self.input_type == 'binary':
                dec_reg = bernoulli_kl(x_mu, x_mu_corr, dim=1)
            else:

                dec_reg = gaus_kl(x_mu.reshape(MB, -1), x_logvar.reshape(MB,-1),
                                  x_mu_corr.reshape(MB,-1), x_logvar_corr.reshape(MB,-1), dim=1)

            dec_reg = dec_reg * weights.to(dec_reg.device)

            # encoder reg: Symmetric KL between 2 gaus mixtures
            sample_correct = self.reparameterize(z_mu_corr, z_logvar_corr)
            pseudoinp = torch.cat(self.prior.mu_list[:N])  # 1 x z_hid
            z_mu, z_logvar = self.q_z(pseudoinp)

            sample_curr = self.reparameterize(z_mu, z_logvar)

            log_q_curr1 = self.log_gaus_mixture(sample_curr, z_mu, z_logvar, weights)
            log_q_correct1 = self.log_gaus_mixture(sample_curr, z_mu_corr, z_logvar_corr, weights)

            log_q_curr2 = self.log_gaus_mixture(sample_correct, z_mu, z_logvar, weights)
            log_q_correct2 = self.log_gaus_mixture(sample_correct, z_mu_corr, z_logvar_corr, weights)

            enc_reg = 0.5*((log_q_curr1 - log_q_correct1) + (log_q_correct2 - log_q_curr2))
            reg = enc_reg + dec_reg
        else:
            reg = torch.zeros(1)
        if init_tr:
            self.train()
        return reg.sum()

    def finish_training_task(self):
        init_tr = self.training
        if init_tr:
            self.eval()
        # self.update_component_weigts()
        self.prior.update_optimal_prior(self.encoder, self.decoder)
        # self.get_component_reconstructions()

        if init_tr:
            self.train()

    def update_component_weigts(self):
        print('Pruning components')
        ## only for current task!!
        curr_task = self.prior.task_weight == self.prior.num_tasks
        w = self.prior.weights[curr_task].clone()
        ps = torch.cat(self.prior.mu_list)[curr_task]
        with torch.no_grad():
            mean_pr, logvar_pr = self.q_z(ps)

        w_new = nn.Parameter(w)
        opt = torch.optim.Adam([w_new], lr=0.0005)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.1)

        N = 1000
        N_comp = ps.shape[0]
        for it in range(int(500)):
            opt.zero_grad()

            #sample from curr prior
            idx = [k for k in np.arange(N_comp) for _ in range(int(N * w_new[k]))]
            z_q_mean = torch.stack([mean_pr[i] for i in idx])
            z_q_logvar = torch.stack([logvar_pr[i] for i in idx])
            z_sample = self.reparameterize(z_q_mean, z_q_logvar)

            # eval log_pr
            log_pr = self.log_gaus_mixture(z_sample, mean_pr[w_new > 0],
                                          logvar_pr[w_new > 0], w_new[w_new > 0])

            # eval log_opt
            www = torch.ones(self.mean_opt.shape[0]) / self.mean_opt.shape[0]
            log_opt = self.log_gaus_mixture(z_sample, self.mean_opt, self.logvar_opt, www)

            kl1 = (log_pr - log_opt).mean()

            #sample from opt prior
            idx = torch.randint(self.mean_opt.shape[0], (N,))
            z_q_mean = torch.stack([self.mean_opt[i] for i in idx])
            z_q_logvar = torch.stack([self.logvar_opt[i] for i in idx])
            z_sample = self.reparameterize(z_q_mean, z_q_logvar)

            # eval log_pr
            log_pr = self.log_gaus_mixture(z_sample, mean_pr[w_new > 0],
                                          logvar_pr[w_new > 0], w_new[w_new > 0])

            # eval log_opt
            log_opt = self.log_gaus_mixture(z_sample, self.mean_opt, self.logvar_opt, www)
            kl2 = (log_opt - log_pr).mean()

            loss = 0.5*kl1 + 0.5*kl2

            loss.backward()
            opt.step()
            sched.step(loss)
            w_new.data = torch.clamp(w_new.data, 0, 1)
        w_new.data = w_new.data/w_new.data.sum()
        self.prior.prune(w_new.data)

    def generate_x(self, N=25):
        N_comp = len(self.prior.pr_q_means)
        idx = np.random.choice(N_comp, size=N, replace=True,
                              p=(self.prior.weights[:N_comp] /
                                 self.prior.task_weight[:N_comp][-1]).cpu().numpy())
        z_q_mean = torch.cat([self.prior.pr_q_means[i] for i in idx])
        z_q_logvar = torch.cat([self.prior.pr_q_logvars[i] for i in idx])
        z_sample = self.reparameterize(z_q_mean, z_q_logvar)

        samples_rand, _ = self.p_x(z_sample)
        return samples_rand

    def calculate_boosting_loss(self, pr_means, pr_logvars, pr_w, lbd=1):
        init_tr = self.training
        if init_tr:
            self.eval()
        # get h params
        z_q_mean, z_q_logvar = self.q_z(self.pseudo_prep(self.h_mu))  # 1 x z_dim

        # z_hat
        z_sample = self.reparameterize(z_q_mean, z_q_logvar)

        entropy = 0.5 * (1 + math.log(2*math.pi) + z_q_logvar).sum() # -log h
        # 1 x Z1
        log_mean_q = self.opt_prior(z_sample)
        log_p_z = self.log_gaus_mixture(z_sample, pr_means, pr_logvars, pr_w) # log p_(t-1)

        loss = -entropy - lbd*log_mean_q + lbd*log_p_z
        if init_tr:
            self.train()
        return loss, entropy, log_mean_q, log_p_z

    def train_component(self, opt, lbd, max_steps):
        history_boost = {'train_loss_boost': 0, 'entropy': 0, 'log_mean_q': 0, 'log_p_z': 0}
        for g in opt.param_groups:
            # g['lr'] = 0.0003
            g['lr'] = 0.003
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.5)
        loss_hist = [1e10]
        mu_prev = self.pseudo_prep(self.h_mu.data).cpu().numpy().copy()
        # current prior keks (cause it does not change)
        with torch.no_grad():
            means, logvars, w = self.get_current_prior(curr_task=True)

        for boost_ep in range(1, max_steps + 1):
            opt.zero_grad()
            loss, entropy, log_mean_q, log_p_z = \
                self.calculate_boosting_loss(means, logvars, w, lbd)
            loss.backward()
            loss_hist.append(loss.item())
            opt.step()
            history_boost['train_loss_boost'] += loss.item()
            history_boost['entropy'] += entropy.item()
            history_boost['log_mean_q'] += log_mean_q.item()
            history_boost['log_p_z'] += log_p_z.item()
            mu_curr = self.pseudo_prep(self.h_mu.data).cpu().numpy().copy()
            pr_norm = np.linalg.norm(mu_prev)
            if np.abs(loss_hist[-1]-loss_hist[-2]) < 1e-2 and boost_ep > 2000:
                print('Component fully trained in {} iterations'.format(boost_ep))
                break
            mu_prev = mu_curr.copy()
            scheduler.step(loss)

        history_boost['train_loss_boost'] /= boost_ep
        history_boost['entropy'] /= boost_ep
        history_boost['log_mean_q'] /= boost_ep
        history_boost['log_p_z'] /= boost_ep
        return history_boost

    def add_component(self, boost_opt,  lbd, boost_steps=30000, from_prev=False, from_input=None):
        """
        Takes current parameters of the distribution h and add them to the list
        of learned parameters
        """
        # and self.prior.num_tasks == 1
        if self.prior.num_comp == 0:
            self.init_prior()
        with torch.no_grad():
            self.mean_opt, self.logvar_opt = self.q_z(self.X_opt)

        w, it = 0, 0
        while w == 0 and it < 1:
            it += 1
            # reset h parameters
            if from_prev:
                # sample = self.prior.sample()
                self.reset_parameters()
            elif from_input is not None:
                self.reset_parameters(from_input)
            else:
                self.reset_parameters()

            # train component
            history_boost = self.train_component(boost_opt, lbd, boost_steps)
            # get the weight
            if self.comp_weight == 'fixed':
                w = 0.5
            elif self.comp_weight == 'grad':
                w = self.get_opt_alpha()
                # w = torch.clamp(w, torch.tensor(.01), torch.tensor(.95))
            else:
                w = None
        print('Trained weight:', w)
        # add component to the prior
        mu = nn.Parameter(self.pseudo_prep(self.h_mu).clone().detach().data, requires_grad=False)
        self.prior.add_component(mu, alpha=w)
        return history_boost

    def reset_parameters(self, mu=None):
        """
        Reset parameters of the h distribution (to start learning a new component)
        """
        if mu is None:
            self.h_mu.data.normal_(self.pseudoinputs_mean, self.pseudoinputs_std)
        else:
            noise = torch.randn([1] + self.input_size).to(mu.device)*0.05
            self.h_mu.data = mu.reshape([1] + self.input_size) + noise

    def get_opt_alpha(self, max_iter=int(1e4), tol=1e-4, lr=5e-1):
        w = torch.tensor(.5)
        trace_w = []
        for i in range(max_iter):
            grad = self.alpha_grad(w)
            w -= (lr / (i + 1.)) * grad
            trace_w.append(w.item())
            w = torch.clamp(w, torch.tensor(1e-4), torch.tensor(1.))
            if (i > 20) and (np.abs(trace_w[-1] - trace_w[-2]) <= tol):
                break
        return w.detach()

    def alpha_grad(self, point):
        init_tr = self.training
        if init_tr:
            self.eval()
        with torch.no_grad():
            z_q_mean, z_q_logvar = self.q_z(self.pseudo_prep(self.h_mu))  # 1 x z_dim
            h_sample = self.reparameterize(z_q_mean, z_q_logvar)
            N = 10
            c = len(self.prior.mu_list)
            w = self.prior.weights / self.prior.task_weight[-1]
            id = np.random.choice(c, size=N, replace=True, p=w.cpu().numpy())

            x = torch.cat([self.prior.mu_list[i] for i in id])
            z_q_mean, z_q_logvar = self.q_z(x)  # 1 x z_dim
            p_sample = self.reparameterize(z_q_mean, z_q_logvar)
            # print(self.grad_weight(h_sample, point), self.grad_weight(p_sample, point))
            grad = self.grad_weight(h_sample, point) - self.grad_weight(p_sample, point)
        if init_tr:
            self.train()
        return grad

    def grad_weight(self, z_sample, alpha):
        with torch.no_grad():
            log_q_z = self.opt_prior(z_sample).mean(0)
            # log_q_z = log_q_z.mean(0)  # (1, )
            z_q_mean, z_q_logvar = self.q_z(self.pseudo_prep(self.h_mu))
            log_p_z = self.log_p_z(z_sample).mean(0)
            log_h_z = log_Normal_diag(z_sample, z_q_mean, z_q_logvar, dim=1).mean(0)

        log_h_z += torch.log(alpha)
        log_p_z += torch.log(1. - alpha)
        # print(log_q_z, log_h_z, log_p_z)
        comb_log_p = torch.logsumexp(torch.cat((log_p_z.reshape(1, ),
                                                log_h_z.reshape(1, )), 0), 0)
        return comb_log_p - log_q_z

    def visualize_results(self, args, dir, epoch, real):
        # initialize folders
        init_tr = self.training
        if init_tr:
            self.eval()
        if not os.path.exists(os.path.join(dir, 'reconstruction')):
            os.makedirs(os.path.join(dir, 'reconstruction'))
            plot_images(args, real.data.cpu().numpy(),
                        os.path.join(dir, 'reconstruction'), 'real', size_x=3, size_y=3)

        if not os.path.exists(os.path.join(dir, 'components')):
            os.makedirs(os.path.join(dir, 'components'))
        if not os.path.exists(os.path.join(dir, 'prior_samples')):
            os.makedirs(os.path.join(dir, 'prior_samples'))

        if epoch == 1:
            plot_images(args, self.X_opt.data.cpu().numpy()[0:36],
                        os.path.join(dir, 'components'), 'real', size_x=6, size_y=6)
        # plot current reconstructions
        if epoch < 10 or epoch%10 == 0:
            x_mean = self.reconstruct_x(real)
            plot_images(args, x_mean.data.cpu().numpy(),
                        os.path.join(dir, 'reconstruction'), str(epoch), size_x=3, size_y=3)

            n_comp = len(self.prior.pr_q_means)
            c = len(self.prior.mu_list)
            n_tsk = str(self.prior.num_tasks)

            # prior components
            mu = self.prior.mu_list
            if c == 0:
                # mu = [self.pseudo_prep(self.h_mu_f).data]
                mu = [self.h_mu_f(self.idle_input).data]
            # print(mu[-100:])
            out = torch.cat(mu[-100:], 0).cpu().numpy()
            # print(out)
            plot_images(args, out, os.path.join(dir, 'components'),
                        str(n_tsk) + '_' + str(epoch), size_x=10, size_y=10)

            # prior samples
            if c > 0:
                z_p_mean, z_p_logvar, _ = self.get_current_prior()
                res = [torch.stack([self.reparameterize(z_p_mean[i], z_p_logvar[i])
                              for _ in range(10)]) for i in range(max(0, c-10), c)]
                out = [self.p_x(res[i])[0].detach().cpu() for i in range(len(res))]
                out = torch.cat(out, 0).numpy()
                plot_images(args, out, os.path.join(dir, 'prior_samples'),
                            str(n_tsk) + '_' + str(epoch),
                            size_x=10, size_y=10)
        if init_tr:
            self.train()
