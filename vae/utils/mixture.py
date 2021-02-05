import torch
import torch.nn as nn
import math
import numpy as np


class MultivariateNormalLRD:
    def __init__(self, mu, sigma, u, init_logdet=True):
        """
        Multivariate gaussian distribution with low rank plus diagonal covariance matrix
        \Sigma = D + uu^T, where D is diagonal

        sigma -- diagonal elements of D (C, z_dim)
        u --- matrix from the low rank part (C, z_dim, r)
        mu --- mean (C, z_dim)
        """
        self.sigma = sigma
        self.u = u
        self.mu = mu
        self.batch_shape, self.N = self.u.shape[:-2], self.u.shape[-2]
        self.r = self.u.shape[-1]

        assert sigma.shape[:-1] == mu.shape[:-1] == self.batch_shape, \
            'different batch dimentions: sigma {}, mu {}, u {}'.format(sigma.shape[:-1],
                                                                       mu.shape[:-1],
                                                                       u.shape[:-2])
        if init_logdet:
            self.K = torch.cholesky(self.capacitance_matrix())
            self.log_det = 2 * torch.log(self.K.diagonal(dim1=-2, dim2=-1)).sum(-1) + \
                       torch.log(self.sigma).sum(-1)  # (batch_size)

    def covariance_matrix(self):
        return torch.diag_embed(self.sigma) + torch.matmul(self.u,
                                                           self.u.transpose(-1, -2))

    def capacitance_matrix(self):
        """
        I + U^T D^{-1} U
        """
        K = torch.matmul(self.u.transpose(-1, -2) / self.sigma.unsqueeze(-2), self.u)
        I = torch.eye(self.r).to(self.mu.device)
        K += I.expand(self.batch_shape + torch.Size([self.r, self.r]))  # add I
        return K

    def entropy(self):
        H = 0.5 * self.N * (1.0 + math.log(2 * math.pi)) + 0.5 * self.log_det
        return H

    def rsample(self, size=torch.Size()):
        # (size x MB x N)
        eps1 = torch.FloatTensor(size + self.batch_shape +
                                 torch.Size([self.N])).normal_().to(self.mu.device)
        # (size x MB x r)
        eps2 = torch.FloatTensor(size + self.batch_shape +
                                 torch.Size([self.r])).normal_().to(self.mu.device)

        # (size x MB x z_dim)
        z = self.sigma.pow(0.5) * eps1 + \
            torch.matmul(self.u, eps2.unsqueeze(-1)).squeeze(-1) + self.mu
        return z

    def log_prob(self, point):
        """
        -0.5 (const + log_det + (x-mu)^T * D^{-1} * (x-mu) - (x-mu)^T*D^{-1}*U * C^{-1} * U^T*D^{-1}*(x-mu) )
        """
        # print(point.device, self.mu.device)
        diff = point - self.mu
        # print('diff', diff.shape)
        const = self.N * math.log(2 * math.pi)

        sum1 = (diff.pow(2) / self.sigma).sum(-1)

        diff_ext = torch.matmul((diff / self.sigma).unsqueeze(-2), self.u).squeeze(-2)

        K_inv = self.inverseL(self.K).transpose(-1, -2)
        sum2 = torch.matmul(diff_ext.unsqueeze(-2), K_inv)
        sum2 = torch.bmm(sum2.view(-1, 1, self.r),
                         sum2.view(-1, 1, self.r).transpose(-1, -2)).view(
            point.shape[:-2] + self.batch_shape)
        return -0.5 * (const + self.log_det + sum1 - sum2)

    @staticmethod
    def inverseL(L):
        """
    https://www.pugetsystems.com/labs/hpc/PyTorch-for-Scientific-Computing---Quantum-Mechanics-Example-Part-3-Code-Optimizations---Batched-Matrix-Operations-Cholesky-Decomposition-and-Inverse-1225/

        """
        n = L.shape[-1]
        invL = torch.zeros_like(L)
        for j in range(0, n):
            invL[..., j, j] = 1.0 / L[..., j, j]
            for i in range(j + 1, n):
                S = 0.0
                for k in range(i + 1):
                    S = S - L[..., i, k] * invL[..., k, j].clone()
                invL[..., i, j] = S / L[..., i, i]
        return invL


class MvnMixture(nn.Module):
    def __init__(self, mu, sigma, u=None, alpha=[1.]):
        super(MvnMixture, self).__init__()
        self.num_comp = len(mu)  # id of the component in the current task
        self.num_tasks = 1  # id of the current task
        self.base_distr = MultivariateNormalLRD

        # [(z_hid), ...]
        self.mu_list = mu
        self.sigma_list = sigma
        # [(z_hid, r), ...]
        self.u_list = u

        # weight of the component in the mixure for a given task
        self.weights = torch.Tensor([alpha[i] for i in range(self.num_comp)])
            # to which task does the component correspond to
        self.task_weight = torch.Tensor([1 for _ in range(self.num_comp)])
        self.full_dist = None

    def log_density(self, point):
        if self.full_dist is None:
            N = len(self.mu_list)
            mu = torch.stack([self.mu_list[i] for i in range(N)])
            sigma = torch.stack([self.sigma_list[i] for i in range(N)])
            u = torch.stack([self.u_list[i] for i in range(N)])
            self.full_dist = self.base_distr(mu, sigma, u)
        point = point.unsqueeze(1)  # (MB x 1 x z_dim)
        log_probs = self.full_dist.log_prob(point)  # MB x comp

        w = self.weights/self.num_tasks  # comp
        log_w = torch.log(w).unsqueeze(0).to(point.device)  # 1 x comp
        log_probs = torch.logsumexp(log_probs + log_w, 1)  # MB x 1
        return log_probs

    def sample(self, n=1):
        mixture_idx = np.random.choice(len(self.mu_list), size=n, replace=True,
                                       p=(self.weights/self.task_weight[-1]).cpu().numpy())
                                       # .numpy())
        mu = torch.stack([self.mu_list[i] for i in mixture_idx])
        sigma = torch.stack([self.sigma_list[i] for i in mixture_idx])
        u = torch.stack([self.u_list[i] for i in mixture_idx])
        return self.base_distr(mu, sigma, u).rsample()

    def add_component(self, mu, sigma, u, alpha=None):
        self.num_comp += 1
        curr_task = self.task_weight == self.num_tasks
        if alpha is None:
            alpha = 1 / self.num_comp
        if sum(curr_task) > 0:
            self.weights[curr_task] *= (1 - alpha)
            self.weights = torch.cat([self.weights.cpu(),
                                      torch.Tensor([alpha]).cpu()])
        else:
            self.weights = torch.cat([self.weights.cpu(), torch.Tensor([1])])
        self.task_weight = torch.cat([self.task_weight.cpu(),
                                      torch.Tensor([self.num_tasks]).cpu()])

        assert np.allclose(self.weights.cpu().sum(), self.num_tasks), \
            'Components\' weights do not sum up to one ' + str(self.weights.sum())
        print(self.weights)

        self.mu_list.append(mu)
        self.sigma_list.append(sigma)
        self.u_list.append(u)

        self.full_dist = None


class VampMixture(nn.Module):
    def __init__(self, pseudoinputs, alpha=[1.]):
        super(VampMixture, self).__init__()
        self.num_comp = len(pseudoinputs)  # id of the component in the current task
        self.num_tasks = 1  # id of the current task
        # self.base_distr = MultivariateNormalLRD

        # [(1, inp_size), ...]
        self.mu_list = pseudoinputs

        # weight of the component in the mixure for a given task
        self.weights = torch.Tensor([alpha[i] for i in range(self.num_comp)])
            # to which task does the component correspond to
        self.task_weight = torch.Tensor([1 for _ in range(self.num_comp)])

        self.pr_q_means = []
        self.pr_q_logvars = []
        self.reconstruction_means = []
        self.reconstruction_logvars = []

    def add_component(self, pseudoinput, alpha=None):
        self.num_comp += 1
        curr_task = self.task_weight == self.num_tasks
        if alpha is None:
            alpha = 1. / self.num_comp
        if sum(curr_task.float()) > 0:
            self.weights[curr_task] *= (1 - alpha)
            self.weights = torch.cat([self.weights.cpu(),
                                      torch.Tensor([alpha]).cpu()])
        else:
            self.weights = torch.cat([self.weights.cpu(), torch.Tensor([1])])

        self.task_weight = torch.cat([self.task_weight.cpu(),
                                      torch.Tensor([self.num_tasks]).cpu()])
        assert np.allclose(self.weights.cpu().sum(), self.num_tasks), \
            'Components\' weights do not sum up to one ' + str(self.weights.sum())
        self.mu_list.append(pseudoinput)

    def prune(self, w_new):
        tot_comp = len(self.mu_list)
        # we update only current task
        curr_task = self.task_weight == self.num_tasks
        assert len(w_new) == sum(curr_task)

        non_zero = w_new > 0
        self.weights = torch.cat([self.weights[~curr_task], w_new[non_zero]])
        self.task_weight = torch.cat([self.task_weight[~curr_task],
                                      self.task_weight[curr_task][non_zero]])

        assert np.allclose(self.weights.cpu().sum(), self.num_tasks), \
            print(self.weights.cpu().sum(), self.num_tasks)

        new_ps = [self.mu_list[i] for i in range(tot_comp-len(w_new))]
        for good_id in np.where(non_zero)[0]:
            idx = np.where(curr_task)[0][good_id]
            new_ps.append(self.mu_list[idx])

        self.mu_list = new_ps
        self.num_comp = float(non_zero.sum())

    def encode_component(self, comp_id, encoder):
        comp_mean = self.mu_list[comp_id]
        with torch.no_grad():
            z_mu, z_logvar = encoder(comp_mean)
        self.pr_q_means.append(nn.Parameter(z_mu.data, requires_grad=False))
        self.pr_q_logvars.append(nn.Parameter(z_logvar.data, requires_grad=False))
        assert len(self.pr_q_means) <= len(self.mu_list)

    def update_optimal_prior(self, encoder, decoder):
        """
        Save q(z|u) for a given task (for further regularization | usage)
        """
        print('Updating q(z|u)....')
        curr_task = (self.task_weight == self.num_tasks).nonzero().squeeze(1)
        for idx in curr_task:
            comp_mean = self.mu_list[idx]  # 1 x z_hid
            with torch.no_grad():
                z_mu, z_logvar = encoder.forward(comp_mean)
                x_mu, x_logvar = decoder.forward(z_mu)

            self.pr_q_means.append(nn.Parameter(z_mu.data, requires_grad=False))  # inp_size
            self.pr_q_logvars.append(nn.Parameter(z_logvar.data, requires_grad=False))
            self.reconstruction_means.append(nn.Parameter(x_mu.data,
                                                          requires_grad=False))
            self.reconstruction_logvars.append(nn.Parameter(x_logvar.data,
                                                            requires_grad=False))