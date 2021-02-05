import math

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from vae.utils.distributions import log_Bernoulli, log_Normal_diag, log_Logistic_256


class SimpleVAE(nn.Module):
    def __init__(self, z1_size, input_type, ll, achitecture):
        super(SimpleVAE, self).__init__()
        self.hid_dim = z1_size
        # self.inp_dim = input_size
        self.input_type = input_type
        self.ll = ll

        self.encoder, self.decoder = achitecture()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or \
                    isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def q_z(self, x):
        """
        Encoder
        :param x: input image
        :return: parameters of q(z|x), (MB, hid_dim)
        """
        z_q_mean, z_q_logvar = self.encoder(x)
        return z_q_mean, z_q_logvar

    def p_x(self, z):
        """
        Decoder
        :param z: latent vector          (MB, hid_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        x_mean, x_logvar = self.decoder(z)
        return x_mean, x_logvar

    def forward(self, x):
        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        # x_mean = p(x|z)
        x_mean, x_logvar = self.p_x(z_q)
        # reshape for convolutional architectures
        x_mean = x_mean.reshape(x_mean.shape[0], -1)
        x_logvar = x_logvar.reshape(x_mean.shape[0], -1)
        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar

    def log_p_z(self, z):
        """
        Prior
        :param z: latent vector     (MB, hid_dim)
        :return: \sum_i log p(z_i)  (1, )
        """
        raise NotImplementedError("To be implemented")

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    def NLL(self, x, x_mean, x_logvar):
        if self.input_type == 'binary':
            nll = -log_Bernoulli(x, x_mean, dim=1)
        elif self.input_type == 'gray' or self.input_type == 'continuous':
            if self.ll == 'normal':
                nll = -log_Normal_diag(x, x_mean, x_logvar, dim=1)
            elif self.ll == 'logistic':
                nll = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
            elif self.ll == 'l1':
                nll = torch.abs(x - x_mean).sum(1)
            else:
                raise Exception('Wrong log likelihood type!')
        else:
            raise Exception('Wrong input type!')
        return nll

    def kl(self, z, z_mean, z_logvar):
        """
        KL-divergence between p(z) and q(z|x)
        :param z:           (MB, hid_dim)
        :param z_mean:      (MB, hid_dim)
        :param z_logvar:    (MB, hid_dim)
        :return: KL         (MB, )
        """
        log_p_z = self.log_p_z(z)
        log_q_z = log_Normal_diag(z, z_mean, z_logvar, dim=1)
        kl_value = log_q_z - log_p_z
        return kl_value
        # raise NotImplementedError("To be implemented")

    def calculate_loss(self, x, beta=1., average=False, eps=0):
        """
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :param eps: Float
        :return: -RE + beta * KL + eps * regularization (MB, ) or (1, )
        """
        MB = x.shape[0]

        # forward
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

        # data term
        nll = self.NLL(x.view(MB, -1), x_mean, x_logvar)
        # KL-divergence
        kl = self.kl(z_q, z_q_mean, z_q_logvar)
        loss = nll + beta * kl

        if average:
            loss = torch.mean(loss, 0)
            nll = torch.mean(nll, 0)
            kl = torch.mean(kl, 0)

        # add regularizartion, if needed
        if eps > 0:
            reg = self.generator_regularization()
            loss += eps * reg
            return loss, -nll, kl, reg.detach().cpu().numpy()
        return loss, -nll, kl, 0

    def calculate_likelihood(self, loader, device, samples=5000):
        """
        Use IS to estimate likelihood
        :param X: dataset, (N, inp_dim)
        :param dir: directory to save the histogram
        :param mode:
        :param samples: Samples per observation
        :param MB: Mini-batch size
        :return:
        """

        # define maximal number of tasks
        max_tsk = loader.dataset.all_tasks.shape[0]

        # likelihood_test = []
        total_ll = []  # save here for final logsumexp
        per_task_ll = []  # separate ll for each task

        for t in range(max_tsk):
            # dataset with images of a proper class only
            loader.dataset.set_task(t)
            print(len(loader), ' batches of data in task ', t)

            # set proper head for encoder/decoder (if relevant)
            if hasattr(self.encoder, 'current_head'):
                self.encoder.current_head = t
                self.decoder.current_head = t

            current_task_lls = []

            for x, _ in loader:
                batch_ll = []
                for _ in range(samples):
                    with torch.no_grad():
                        x = x.to(device)
                        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
                        # -RE
                        NLL = self.NLL(x.reshape(x_mean.shape), x_mean, x_logvar)

                        # NLL = self.NLL(x.view(samples, -1), x_mean.view(samples, -1),
                        #                x_logvar.view(samples, -1))
                        log_p_z = self.log_p_z(z_q)
                        log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
                        a_tmp = - NLL + log_p_z - log_q_z
                    batch_ll.append(a_tmp.cpu().data)

                ll = torch.logsumexp(torch.stack(batch_ll, 0), 0) - math.log(samples)
                current_task_lls.append(ll)

            per_task_ll.append(-torch.cat(current_task_lls).mean())
            total_ll.append(torch.cat(current_task_lls))
        return per_task_ll, -torch.cat(total_ll).mean()

    def calculate_lower_bound(self, X_full, MB=500):
        lower_bound = 0.
        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB]
            with torch.no_grad():
                loss, _, _, _ = self.calculate_loss(x, average=True)
            lower_bound += loss.cpu().item()
        lower_bound /= I
        return lower_bound

    def generate_x(self, N=25):
        raise NotImplementedError("Implement method for sampling x from prior")

    def visualize_results(self):
        raise NotImplementedError("Implement method for sampling x from prior")

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)
