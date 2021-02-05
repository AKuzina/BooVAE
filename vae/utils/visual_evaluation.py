import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.switch_backend('agg')


def plot_histogram(x, dir, mode):
    fig = plt.figure()
    # the histogram of the data
    n, bins, patches = plt.hist(x, 100, normed=True, facecolor='blue', alpha=0.5)

    plt.xlabel('Log-likelihood value')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.savefig(os.path.join(dir, 'histogram_' + mode + '.png'), bbox_inches='tight')
    plt.close(fig)


def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):
    fig = plt.figure(figsize=(size_x, size_y))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape(
            (args.input_size[0], args.input_size[1], args.input_size[2]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if args.input_type == 'binary' or args.input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')
        else:
            plt.imshow((sample - np.min(sample)) / np.ptp(sample))

    plt.savefig(os.path.join(dir, file_name + '.png'), bbox_inches='tight')
    plt.close(fig)
