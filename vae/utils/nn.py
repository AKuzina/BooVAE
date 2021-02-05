import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from vae.utils.bayes import BayesLinear

# LAYERS
class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None, bayes=False):
        super(NonLinear, self).__init__()

        self.activation = activation
        if bayes:
            self.linear = BayesLinear(int(input_size), int(output_size), bias=bias)
        else:
            self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h