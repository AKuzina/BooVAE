import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from vae.utils.visual_evaluation import plot_images
import wandb


def test_vae(args, loader):
    # load model
    model = torch.load(os.path.join(args.dir, 'model.pth'),
                       map_location=torch.device(args.device))
    model.eval()
    N_steps = len(loader)
    history_elbo = {'test_loss': 0, 'test_kl': 0}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            x = data.to(args.device)
            # calculate loss function
            loss, RE, KL, reg = model.calculate_loss(x, beta=args.MAX_BETA, average=True,
                                                     eps=args.eps_gen)

            hist = {'test_loss': loss.item(), 'test_kl': KL.item()}
            history_elbo = {k: history_elbo[k] + hist[k] / N_steps for k in
                            set(history_elbo)}
        # save elbo stats
        wandb.log(history_elbo)

        plot_data = x[:25].to(args.device)
        plot_lbls = target[:25]

        if len(args.input_size) > 2:
            # VISUALIZATION: plot real images
            plot_images(args, plot_data.data.cpu().numpy(), args.dir, 'real', size_x=5, size_y=5)

            # VISUALIZATION: plot reconstructions
            if args.multihead:
                samples = []
                for x, y in zip(plot_data,  plot_lbls):
                    model.encoder.current_head = int(y)
                    model.decoder.current_head = int(y)
                    samples.append(model.reconstruct_x(x.unsqueeze(0)))
                samples = torch.stack(samples)
                print(samples.shape)
            else:
                samples = model.reconstruct_x(plot_data)

            plot_images(args, samples.data.cpu().numpy(), args.dir, 'reconstructions',
                        size_x=5, size_y=5)

            # VISUALIZATION: plot generations
            if args.multihead:
                samples_rand = model.generate_multihead(25)
                for i in range(len(samples_rand)):
                    plot_images(args, samples_rand[i].data.cpu().numpy(), args.dir,
                                '{}_generations'.format(i), size_x=5, size_y=5)
            else:
                samples_rand = model.generate_x(25)
                plot_images(args, samples_rand.data.cpu().numpy(), args.dir, 'generations',
                        size_x=5, size_y=5)

            if 'boost' in args.prior:
                # VISUALIZE pseudoinputs
                # prior components
                mu = model.prior.mu_list
                out = torch.cat(mu[-100:], 0).cpu().numpy()
                plot_images(args, out, args.dir, 'prior_components', size_x=10, size_y=10)

        # CALCULATE test log-likelihood
        t_ll_s = time.time()
        per_task_ll, nll_test = model.calculate_likelihood(loader, args.device, samples=args.S)
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(nll_test, t_ll_e - t_ll_s))
        wandb.log({'Test_nll': nll_test})
        for t in range(len(per_task_ll)):
            wandb.log({'Test_nll_task_{}'.format(t): -per_task_ll[t]})