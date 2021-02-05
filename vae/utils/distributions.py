import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

min_epsilon = 1e-5
max_epsilon = 1. - 1e-5


def gaus_kl(q_mu, q_logsigmasq, p_mu, p_logsigmasq, dim=1):
    """

    Compute KL-divergence KL(q || p) between n pairs of Gaussians
    with diagonal covariational matrices.
    Do not divide KL-divergence by the dimensionality of the latent space.

    Input: q_mu, p_mu, Tensor of shape n x d - mean vectors for n Gaussians.
    Input: q_sigma, p_sigma, Tensor of shape n x d - standard deviation
           vectors for n Gaussians.
    Return: Tensor of shape n - each component is KL-divergence between
            a corresponding pair of Gaussians.
    """
    res = p_logsigmasq - q_logsigmasq - 1 + torch.exp(q_logsigmasq - p_logsigmasq)
    res = res + (q_mu - p_mu).pow(2) / (torch.exp(p_logsigmasq) + 1e-5)
    if dim is not None:
        return 0.5 * res.sum(dim=dim)
    else:
        return 0.5 * res


def bernoulli_kl(q_mu, p_mu, dim=1):
    res = q_mu * (torch.log(q_mu + 1e-5) - torch.log(p_mu + 1e-5))
    res += (1 - q_mu) * (torch.log(1 - q_mu + 1e-5) - torch.log(1 - p_mu + 1e-5))
    if dim is not None:
        return res.sum(dim=dim)
    else:
        return res


def log_Normal_diag(x, mean, log_var, dim=None):  # average=False,
    log_normal = -0.5 * (math.log(2.0 * math.pi) + log_var + torch.pow(x - mean, 2) / (
        torch.exp(log_var)) + 1e-5)
    if dim is not None:
        return log_normal.sum(dim=dim)
    else:
        return log_normal
    # if average:
    #     return torch.mean(log_normal, dim)
    # else:
    #     return torch.sum(log_normal, dim)


def log_Normal_standard(x, dim=None):  # average=False,
    log_normal = -0.5 * (math.log(2.0 * math.pi) + torch.pow(x, 2))
    if dim is not None:
        return log_normal.sum(dim=dim)
    else:
        return log_normal
    # if average:
    #     return torch.mean(log_normal, dim)
    # else:
    #     return torch.sum(log_normal, dim)


def log_Bernoulli(x, mean, dim=None):  # average=False,
    probs = torch.clamp(mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if dim is not None:
        return log_bernoulli.sum(dim=dim)
    else:
        return log_bernoulli
    # if average:
    #     return torch.mean(log_bernoulli, dim)
    # else:
    #     return torch.sum(log_bernoulli, dim)


def log_Logistic_256(x, mean, logvar, dim=None):  # average=False,
    """
    Logistic LogLikelihood
    :param x:
    :param mean:
    :param logvar:
    :param dim:
    :return:
    """
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size / scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = torch.log(cdf_plus - cdf_minus + 1.e-7)

    if dim is not None:
        # if average:
        #     return torch.mean(log_logist_256, dim)
        # else:
        return log_logist_256.sum(dim=dim)
    else:
        return log_logist_256
