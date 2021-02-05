import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.1)
        self.logvar.data.fill_(-12)
        if self.bias is not None:
            self.bias_logvar.data.fill_(-12)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.training:
            w_sample = self.sample(self.weight, self.logvar)
            bias_sample = None
            if self.bias is not None:
                bias_sample = self.sample(self.bias, self.bias_logvar)
            out = F.linear(input, w_sample, bias_sample)
        else:
            out = F.linear(input, self.weight, self.bias)
        return out

    def sample(self, mu, logvar):
        eps = mu.data.new(mu.size()).normal_()
        sample = mu + eps.mul(torch.exp(0.5*logvar))
        return sample


class _BayesConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_BayesConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.mu_weight = nn.Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size))
            self.logsigma_weight = nn.Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.mu_weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size))
            self.logsigma_weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
        self.register_parameter('logsigma_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.mu_weight.data.normal_(0, 0.02)
        self.logsigma_weight.data.fill_(-5)
        if self.mu_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.mu_bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.mu_bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class BayesConv2d(_BayesConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, False, _pair(0), groups,
                                          bias)

    def forward(self, input):
        if self.training:
            sigma_sq = F.conv2d(input.pow(2), self.logsigma_weight.exp(), None,
                                self.stride, self.padding, self.dilation, self.groups)
            sigma_out = torch.sqrt(1e-10 + sigma_sq)

            mu_out = F.conv2d(input, self.mu_weight, self.mu_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            eps = sigma_out.data.new(sigma_out.size()).normal_()
            out = eps.mul(sigma_out) + mu_out
        else:
            out = F.conv2d(input, self.mu_weight, self.mu_bias, self.stride,
                           self.padding, self.dilation, self.groups)
        return out