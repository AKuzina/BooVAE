import math

import numpy as np
import torch
import torch.nn as nn
from vae.utils.nn import NonLinear
from vae.utils.bayes import BayesLinear, BayesConv2d

class EncoderMnist(nn.Module):
    def __init__(self, h_dim, inp_dim, bayes=False):
        super(EncoderMnist, self).__init__()
        self.q_z_layers = nn.Sequential(
            NonLinear(inp_dim, 300, activation=nn.LeakyReLU(inplace=True), bayes=bayes),
            NonLinear(300, 300, activation=nn.LeakyReLU(inplace=True), bayes=bayes)
        )
        self.q_z_mean = NonLinear(300, h_dim, activation=None, bayes=bayes)
        self.q_z_logvar = NonLinear(300, h_dim,
                                    activation=nn.Hardtanh(min_val=-6., max_val=2.),
                                    bayes=bayes)

    def forward(self, x):
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar


class DecoderMnist(nn.Module):
    def __init__(self, h_dim, inp_dim, bayes=False):
        super(DecoderMnist, self).__init__()

        self.p_x_layers = nn.Sequential(
            NonLinear(h_dim, 300, activation=nn.LeakyReLU(inplace=True), bayes=bayes),
            NonLinear(300, 300, activation=nn.LeakyReLU(inplace=True), bayes=bayes)
        )
        self.p_x_mean = NonLinear(300, inp_dim, activation=nn.Sigmoid(), bayes=bayes)

    def forward(self, x):
        z = self.p_x_layers(x)
        x_mean = self.p_x_mean(z)
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar


class EncoderFaMnist(nn.Module):
    def __init__(self, h_dim, inp_dim, internal_dim=1024, bayes=False):
        super(EncoderFaMnist, self).__init__()

        self.q_z_layers = nn.Sequential(
            NonLinear(inp_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes),
            NonLinear(internal_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes)
        )

        self.q_z_mean = NonLinear(internal_dim, h_dim, activation=None, bayes=bayes)
        # self.q_z_logvar = nn.Linear(internal_dim, h_dim)
        self.q_z_logvar = NonLinear(internal_dim, h_dim,
                                    activation=nn.Hardtanh(min_val=-4., max_val=4.),
                                    bayes=bayes)

    def forward(self, x):
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar


class DecoderFaMnist(nn.Module):
    def __init__(self, h_dim, inp_dim, internal_dim=1024, bayes=False):
        super(DecoderFaMnist, self).__init__()

        self.p_x_layers = nn.Sequential(
            NonLinear(h_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes),
            NonLinear(internal_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes)
        )
        self.p_x_mean = NonLinear(internal_dim, inp_dim, activation=nn.Sigmoid(),
                                  bayes=bayes)

    def forward(self, x):
        z = self.p_x_layers(x)
        x_mean = self.p_x_mean(z)
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar


class EncoderNotMnist(nn.Module):
    def __init__(self, h_dim, inp_dim, internal_dim=1024, bayes=False):
        super(EncoderNotMnist, self).__init__()

        self.q_z_layers = nn.Sequential(
            NonLinear(inp_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes),
            NonLinear(internal_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes)
        )

        self.q_z_mean = NonLinear(internal_dim, h_dim, activation=None, bayes=bayes)
        self.q_z_logvar = NonLinear(internal_dim, h_dim,
                                    activation=nn.Hardtanh(min_val=-4., max_val=4.),
                                    bayes=bayes)

    def forward(self, x):
        x = self.q_z_layers(x)
        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar


class DecoderNotMnist(nn.Module):
    def __init__(self, h_dim, inp_dim, internal_dim=1024, bayes=False):
        super(DecoderNotMnist, self).__init__()

        self.p_x_layers = nn.Sequential(
            NonLinear(h_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes),
            NonLinear(internal_dim, internal_dim, activation=nn.LeakyReLU(inplace=True),
                      bayes=bayes)
        )
        self.p_x_mean = NonLinear(internal_dim, inp_dim, activation=nn.Sigmoid(),
                                  bayes=bayes)

    def forward(self, x):
        z = self.p_x_layers(x)
        x_mean = self.p_x_mean(z)
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar


class EncoderCelebA(nn.Module):
    def __init__(self, h_dim=128, k=3, base=5, bayes=False, **kwargs):
        super(EncoderCelebA, self).__init__()
        if bayes:
            conv = BayesConv2d
        else:
            conv = nn.Conv2d
        self.h_dim = h_dim
        self.blocks = []
        self.k = k  ## 3
        self.base = base
        for i in range(4):
            if i == 0:
                block = [conv(3, 2 ** self.base, kernel_size=5, stride=2,
                              padding=1, bias=True),
                         nn.BatchNorm2d(2 ** self.base, affine=True),
                         nn.ReLU(inplace=True)]
            else:
                block = [conv(2 ** (i + self.base - 1), 2 ** (i + self.base),
                                   kernel_size=5, stride=2, padding=1, bias=True),
                         nn.BatchNorm2d(2 ** (i + self.base), affine=True),
                         nn.ReLU(inplace=True)]
            self.blocks.extend(block)
        self.blocks = nn.Sequential(*self.blocks)

        self.fc_mu = nn.Linear(self.k * self.k * 2 ** (self.base + 3), h_dim)
        self.fc_logs2 = NonLinear(self.k * self.k * 2 ** (self.base + 3), h_dim,
                                  activation=nn.Hardtanh(min_val=-3., max_val=4.), bayes=bayes)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view([-1, self.k * self.k * 2 ** (self.base + 3)])
        mu, logs2 = self.fc_mu(x), self.fc_logs2(x)
        return mu, logs2


class DecoderCelebA(nn.Module):
    def __init__(self, h_dim=128, k=8, base=5, **kwargs):
        super(DecoderCelebA, self).__init__()

        self.h_dim = h_dim
        self.k = k  ## 8
        self.base = base
        self.fc1 = nn.Linear(h_dim, self.k ** 2 * 2 ** (self.base + 3), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.blocks = []
        for i in range(3):
            block = [
                nn.ConvTranspose2d(2 ** (self.base + 3 - i), 2 ** (self.base + 2 - i),
                                   kernel_size=5,
                                   stride=2, padding=2, output_padding=1, bias=True),
                nn.BatchNorm2d(2 ** (self.base + 2 - i), affine=True),
                nn.ReLU(inplace=True)]
            self.blocks.extend(block)
        self.blocks = nn.Sequential(*self.blocks)
        self.conv4 = nn.Sequential(nn.Conv2d(2 ** self.base, 3, kernel_size=1,
                                             stride=1, padding=0, bias=True),
                                   nn.Softsign())

    def forward(self, z):
        z = self.fc1(z)
        z = self.relu1(z)
        z = z.view([-1, 2 ** (self.base + 3), self.k, self.k])
        z = self.blocks(z)
        x_mu = self.conv4(z)
        x_logvar = torch.zeros_like(x_mu) - math.log(2.0 * math.pi)
        return x_mu, x_logvar


class MultiheadEncoder(nn.Module):
    def __init__(self, base_encoder, encoder_params, n_heads=0):
        super(MultiheadEncoder, self).__init__()
        """
        :param base_encoder: contructot of encoder network
        :param encoder_params: dictionary of the encoder parameters
        :param n_heads: int, initial number of heads
        """
        self.base_encoder = base_encoder
        self.encoder_params = encoder_params
        self.heads = nn.ModuleList(
            [base_encoder(**encoder_params) for _ in range(n_heads)])
        self.current_head = n_heads - 1

    def forward(self, x):
        mu, logvar = self.heads[self.current_head](x)
        return mu, logvar

    def add_head(self):
        self.heads.append(self.base_encoder(**self.encoder_params))


class MultiheadDecoder(nn.Module):
    def __init__(self, base_decoder, decoder_params, hid_dim, n_heads=0):
        super(MultiheadDecoder, self).__init__()
        self.h_dim = hid_dim
        self.shared_dim = decoder_params['h_dim']
        self.shared_decoder = base_decoder(**decoder_params)
        self.heads = nn.ModuleList(
            [nn.Linear(self.h_dim, self.shared_dim) for _ in range(n_heads)])
        self.current_head = n_heads - 1

    def forward(self, z):
        z = self.heads[self.current_head](z)
        return self.shared_decoder(z)

    def add_head(self):
        self.heads.append(nn.Linear(self.h_dim, self.shared_dim))


def get_architecture(args):
    params_enc = {'h_dim': args.z1_size, 'inp_dim': np.prod(args.input_size),
                  'bayes': args.bayes}
    params_dec = {'h_dim': args.z1_size, 'inp_dim': np.prod(args.input_size),
                  'bayes': args.bayes}
    if args.multihead:
        params_dec['h_dim'] = args.z2_size


    if args.dataset_name == 'mnist':
        enc = EncoderMnist
        dec = DecoderMnist

    elif args.dataset_name == 'notmnist':
        enc = EncoderNotMnist
        dec = DecoderNotMnist

    elif args.dataset_name == 'fashion_mnist':
        enc = EncoderFaMnist
        dec = DecoderFaMnist

    elif args.dataset_name in ['celeba']:
        dec = DecoderCelebA
        enc = EncoderCelebA
        if args.input_size[1] == 32:
            params_enc['k'] = 1
            params_dec['k'] = 4
        elif args.input_size[1] == 64:
            params_enc['k'] = 3
            params_dec['k'] = 8

    if args.multihead:
        enc_m = MultiheadEncoder(enc, params_enc, n_heads=0)
        dec_m = MultiheadDecoder(dec, params_dec, args.z1_size, n_heads=0)
        arc_getter = lambda: (enc_m, dec_m)
    else:
        arc_getter = lambda: (enc(**params_enc), dec(**params_dec))
    return arc_getter, args
