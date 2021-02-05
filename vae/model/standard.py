import torch
import os
from vae.model.simple_vae import SimpleVAE
from vae.utils.distributions import log_Normal_standard, gaus_kl
from vae.utils.visual_evaluation import plot_images


class Simple(SimpleVAE):
    def __init__(self, z1_size, input_type, ll, arc, **kwargs):
        super(Simple, self).__init__(z1_size, input_type, ll, arc)

    def log_p_z(self, z):
        log_prior = log_Normal_standard(z, dim=1)
        return log_prior

    def generate_x(self, N=25):
        device = list(self.parameters())[0].device
        z_sample_rand = torch.FloatTensor(N, self.hid_dim).normal_().to(device)
        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand

    def visualize_results(self, args, dir, epoch, real):
        if not os.path.exists(os.path.join(dir, 'reconstruction')):
            os.makedirs(os.path.join(dir, 'reconstruction'))
        if epoch < 10:
            plot_images(args, real.data.cpu().numpy(),
                    os.path.join(dir, 'reconstruction'), 'real', size_x=3, size_y=3)

        if not os.path.exists(os.path.join(dir, 'prior_samples')):
            os.makedirs(os.path.join(dir, 'prior_samples'))

        if epoch%5 == 0:
            # plot current reconstructions and samples from prior
            x_mean = self.reconstruct_x(real)
            samples = self.generate_x(N=9)
            plot_images(args, x_mean.data.cpu().numpy(),
                        os.path.join(dir, 'reconstruction'), str(epoch), size_x=3, size_y=3)
            plot_images(args, samples.data.cpu().numpy(),
                        os.path.join(dir, 'prior_samples'), str(epoch), size_x=3, size_y=3)
