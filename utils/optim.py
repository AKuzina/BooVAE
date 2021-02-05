import numpy as np
import torch
from torch import autograd
from torch.utils.data import DataLoader
from vae.utils.bayes import BayesLinear, _BayesConvNd
from vae.utils.distributions import gaus_kl


class VCLOptimizer:
    def __init__(self, base_optimizer, model, weight):
        super(VCLOptimizer, self).__init__()
        self.optim = base_optimizer

        self.prior_means = []
        self.prior_logvars = []
        self.names = []
        for name, m in model.named_modules():
            if isinstance(m, BayesLinear) or isinstance(m, _BayesConvNd):
                self.prior_means.append([torch.zeros_like(m.weight.data),
                                         torch.zeros_like(m.bias.data)])

                self.prior_logvars.append([torch.zeros_like(m.logvar.data),
                                           torch.zeros_like(m.bias_logvar.data)])
                self.names.append(name)

        self.weight = weight
        #sanity print
        print('Add prior to', self.names)

    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def new_task(self, dataset, batch_size, num_batches, model, device):
        for m in model.named_modules():
            if m[0] in self.names:
                id = np.where(np.array(self.names) == m[0])[0][0]
                self.prior_means[id] = [m[1].weight.detach().clone(),
                                        m[1].bias.detach().clone()]
                self.prior_logvars[id] = [m[1].logvar.detach().clone(),
                                          m[1].bias_logvar.detach().clone()]

    def wrap_loss(self, loss, model):
        """
        Add VCL regularization (kl with prior)
        :return: loss + vcl_reg
        """
        kl = 0.
        for m in model.named_modules():
            if m[0] in self.names:
                id = np.where(np.array(self.names) == m[0])[0][0]
                w, b = self.prior_means[id]
                w_var, b_var = self.prior_logvars[id]

                curr_mean = m[1].weight
                curr_logvar = m[1].logvar
                kl += gaus_kl(curr_mean, curr_logvar, w, w_var, None).sum()

                curr_mean = m[1].bias
                curr_logvar = m[1].bias_logvar
                kl += gaus_kl(curr_mean, curr_logvar, b, b_var, None).sum()

        return loss + kl*self.weight


class EWCOptimizier:
    def __init__(self, base_optimizer, lbd):
        super(EWCOptimizier, self).__init__()
        self.optim = base_optimizer
        self.estimated_mean = []
        self.estimated_fisher = []
        self.lbd = lbd

    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def _update_mean_params(self, model):
        names = ['h_mu', 'h_logsigma', 'h_u', 'h_mu_f']
        for name, param in model.named_parameters():
            if param.requires_grad and name not in names and 'heads' not in name:
                self.estimated_mean.append(param.data.clone())

    def _update_fisher_params(self, current_ds, batch_size, num_batch,
                              model, device='cpu'):
        names = ['h_mu', 'h_logsigma', 'h_u', 'h_mu_f']
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (inp, target) in enumerate(dl):
            if i > num_batch:
                break
            inp = inp.to(device)
            MB = inp.shape[0]
            x_mean, x_logvar, _, _, _ = model.forward(inp)
            log_liklihoods.append(-model.NLL(inp.view(MB, -1), x_mean, x_logvar))
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood,
                                           list(p[1] for p in model.named_parameters() if
                                                (p[0] not in names
                                                 and 'heads' not in p[0]
                                                 and p[1].requires_grad)))
        for grad in grad_log_liklihood:
            if grad is None:
                self.estimated_fisher.append(torch.tensor(0))
            else:
                self.estimated_fisher.append(grad.data.detach().clone() ** 2)

    def new_task(self, dataset, batch_size, num_batches, model, device):
        self._update_fisher_params(dataset, batch_size, num_batches, model, device)
        self._update_mean_params(model)
        self.optim.zero_grad()

    def wrap_loss(self, loss, model):
        """
        Add EWC regularization
        :return: loss + lbd/2 * ewc_reg
        """
        names = ['h_mu', 'h_logsigma', 'h_u', 'h_mu_f']
        losses = []
        params = list(p[1] for p in model.named_parameters() if
                      (p[0] not in names
                       and 'heads' not in p[0]
                       and p[1].requires_grad
                       ))
        param_names = list(p[0] for p in model.named_parameters() if
                      (p[0] not in names
                       and 'heads' not in p[0]
                       and p[1].requires_grad
                       ))
        for n, param, mean, fisher in zip(param_names, params, self.estimated_mean, self.estimated_fisher):
            # print(n, fisher.shape, param.shape, mean.shape)
            losses.append((fisher * (param - mean) ** 2).sum())
        reg = (self.lbd / 2) * sum(losses)
        return loss + reg
