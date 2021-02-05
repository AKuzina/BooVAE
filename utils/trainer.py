import torch
import torch.nn as nn
import math
import time
import os
import numpy as np
import wandb

import utils.optim as reg_optim


def get_optimizer(args, model):
    # OPTIMIZER
    pat = int(args.early_stopping_epochs * 0.6)
    if 'boost' in args.prior:
        names = ['h_mu']
        param = [model.h_mu]
        boost_optimizer = torch.optim.Adam(param, lr=1e-2)
    else:
        names = []
        boost_optimizer = None

    # wrap optimizer to compute statistics
    if args.regularizer:
        opt = torch.optim.Adam(list(p[1] for p in model.named_parameters() if
                                    p[0] not in names and 'heads' not in p[0]),
                               lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                               patience=pat, verbose=True)
        if args.resume:
            optimizer = torch.load(os.path.join(args.dir, 'optimizer.pth'),
                             map_location=torch.device(args.device))
            
            optimizer.optim = opt
        elif args.incr_resume and args.max_tasks > 1:
            optimizer = torch.load(os.path.join(args.dir, 'optimizer.pth'),
                                   map_location=torch.device(args.device))
            optimizer.optim = opt
            optimizer.lbd = args.reg_weight
        elif args.reg_type == 'ewc':
            optimizer = reg_optim.EWCOptimizier(opt, args.reg_weight)
        elif args.reg_type == 'vcl':
            optimizer = reg_optim.VCLOptimizer(opt, model,
                                               args.reg_weight / args.batch_size)
    else:
        optimizer = torch.optim.Adam(list(p[1] for p in model.named_parameters() if
                                          p[0] not in names and 'heads' not in p[0]),
                                     lr=args.lr,
                                     weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                               patience=pat, verbose=True)
    return optimizer, scheduler, boost_optimizer


def prepare_head(args, model, optimizer, scheduler):
    model.encoder.add_head()
    model.decoder.add_head()
    model.encoder.current_head += 1
    model.decoder.current_head += 1
    print('{} heads in the model'.format(model.encoder.current_head + 1))

    for i in range(model.encoder.current_head):
        print('Freeze head {}'.format(i))
        model.encoder.freeze_head(i)
        model.decoder.freeze_head(i)

    model.to(args.device)
    curr_groups = optimizer.param_groups

    if len(curr_groups) > 1:
        print('Remove {} parameters from optimizer'.format(len(curr_groups[-1]['params'])))
        optimizer.param_groups.pop(-1)

    elif scheduler is not None:
        scheduler.min_lrs = scheduler.min_lrs + scheduler.min_lrs

    head_name = 'heads.{}'.format(model.encoder.current_head)
    print('Add parameters to optimizer:', [p[0] for p in model.named_parameters() if head_name in p[0]])
    optimizer.add_param_group(
            {'params': [p[1] for p in model.named_parameters() if head_name in p[0]]}
        )


def add_self_replay(args, loader, num_task):
    print('Generate samples from ', args.prev_model_path)
    model = torch.load(args.prev_model_path,
                       map_location=torch.device(args.device))
    # sample
    N = num_task*args.self_replay
    Xs = model.generate_x(N)
    ys = torch.tensor([-1 for _ in range(N)])
    # add samples to dataset
    loader.dataset.add_data(Xs.cpu().data, ys)
    return loader


def train_vae(args, train_loader, val_loader, model, optimizer, scheduler, boost_opt):
    # SAVING
    torch.save(args, os.path.join(args.dir, 'config.pth'))

    best_loss = 100000.
    e = 0
    for num_task, task_id in enumerate(args.all_tasks, args.min_task):
        if args.incremental:
            train_loader.dataset.set_task(num_task)
            val_loader.dataset.set_task(num_task)
            train_loader.dataset.add_coreset(args.coreset_size)

            if args.multihead:
                if args.regularizer:
                    prepare_heads(args, model, optimizer.optim, scheduler)
                else:
                    prepare_heads(args, model, optimizer, scheduler)
            if num_task > 0:
                if 'boost' in args.prior:
                    model.X_opt = args.X[num_task]
                    model.X_opt = model.X_opt.to(args.device)
                    model.prior.num_tasks = num_task+1
                    model.prior.num_comp = 0
                    ## add first component
                    mean_opt = model.X_opt.mean(0, keepdim=True)
                    model.prior.add_component(mean_opt)
                if args.self_replay > 0:
                    train_loader = add_self_replay(args, train_loader, num_task)

        for epoch in range(1, args.epochs + 1):
            time_start = time.time()
            history_elbo = train_epoch(args, epoch, model, train_loader, optimizer, boost_opt)
            history_val = validation_epoch(args, epoch, model, val_loader, task_id)
            if scheduler is not None:
                scheduler.step(history_val['val_kl'] + history_val['val_nll'])
            val_loss = args.MAX_BETA * history_val['val_kl'] + history_val['val_nll']

            time_end = time.time()
            time_elapsed = time_end - time_start
            
            # save stats to wandb
            history_elbo['epoch'] = epoch + 1
            history_val['epoch'] = epoch + 1
            wandb.log(history_elbo)
            wandb.log(history_val)

            # printing results
            if 'boost' in args.prior:
                comp = len(model.prior.mu_list)
                num_comp = args.number_components
            else:
                comp, num_comp = 1, 1
            print('Task {}/{}, Epoch: {}/{}, Component: {}/{} Time elapsed: {:.2f}s\n'
                  '* Train loss: {:.2f}  || Val.  loss: {:.2f} \n'
                  '--> Early stopping: {}/{} (BEST: {:.2f})\n'.format(
                num_task + 1, args.all_tasks.shape[0], epoch, args.epochs, comp, num_comp, 
                time_elapsed, history_elbo['train_loss'], val_loss,
                e, args.early_stopping_epochs, best_loss))

            # early-stopping
            if val_loss < best_loss:
                e = 0
                best_loss = val_loss
                print('->model saved<-')
                torch.save(model.to('cpu'), os.path.join(args.dir, 'model.pth'))
                model.to(args.device)
            elif epoch > args.warmup:
                e += 1
                if e > args.early_stopping_epochs or math.isnan(val_loss):
                    break

        wandb.run.summary['val_loss'] = best_loss
        # finish training task
        if 'boost' in args.prior:
            model = torch.load(os.path.join(args.dir, 'model.pth'), 
                               map_location=torch.device(args.device))
            model.finish_training_task()
            torch.save(model.cpu(), os.path.join(args.dir, 'model.pth'))
            model.to(args.device)

        if args.regularizer:
            model = torch.load(os.path.join(args.dir, 'model.pth'),
                               map_location=torch.device(args.device))
            train_loader.dataset.set_task(num_task)
            curr_dset = torch.utils.data.TensorDataset(train_loader.dataset.X, train_loader.dataset.y)
            print('Update regularization term...')
            optimizer.new_task(dataset=curr_dset, batch_size=150,
                               num_batches=100, model=model, device=args.device)
            torch.save(optimizer, os.path.join(args.dir, 'optimizer.pth'))


def train_epoch(args, epoch, model, loader, optimizer, boost_opt):
    history_elbo = {'train_loss': 0, 'train_nll': 0, 'train_kl': 0, 'train_reg':0}
    N_steps = len(loader)
    model.train()
    beta = np.clip(epoch / args.warmup, 0, 1) * args.MAX_BETA
    print('beta: {}'.format(beta))
    if args.regularizer:
        lr = optimizer.optim.param_groups[0]["lr"]
    else:
        lr = optimizer.param_groups[0]["lr"]
    wandb.log({"lr": lr, 'beta': beta, 'epoch': epoch+1})

    for batch_idx, (data, target) in enumerate(loader):
        x = data.to(args.device)
        optimizer.zero_grad()

        #calculate VAE Loss
        loss, RE, KL, reg = model.calculate_loss(x, beta, average=True,
                                                 eps=args.eps_gen)
        if args.regularizer:
            loss = optimizer.wrap_loss(loss=loss, model=model)
        loss.backward()
        optimizer.step()

        global_step = batch_idx + 1 + N_steps*(epoch-1)
        hist = {'train_loss': loss.item(), 'train_nll': -RE.item(), 'train_kl': KL.item(),
                'train_reg': reg, '': global_step}
        wandb.log({k+'_step': hist[k] for k in set(hist)})
        history_elbo = {k: history_elbo[k] + hist[k]/N_steps for k in set(history_elbo)}
        
        if 'boost' in args.prior and epoch <= args.comp_ep*args.number_components+1:
            comp = model.prior.num_comp
            if global_step % int(args.comp_ep*N_steps) == 0 and batch_idx > 0 and comp < args.number_components:
                history_boost = model.add_component(boost_opt, args.lbd, from_input=None)
                history_boost['component'] = comp
                wandb.log(history_boost)

    if 'boost' in args.prior and args.prune:
        comp = model.prior.num_comp
        thr = int(args.comp_ep*args.number_components*2)
        if epoch%thr == 0 and epoch < args.comp_ep*args.number_components*3 and comp > 1:
            model.update_component_weigts()

    return history_elbo


def validation_epoch(args, epoch, model, loader, task_id):
    history_elbo = {'val_loss': 0, 'val_nll': 0, 'val_kl': 0, 'val_reg': 0}
    # set model to evaluation mode
    model.eval()
    N_steps = len(loader)
    # evaluate
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if args.incremental:
                mask = np.isin(target, torch.LongTensor([task_id]))
                data = data[np.where(mask)[0]]
            x = data.to(args.device)

            # calculate loss function
            loss, RE, KL, reg = model.calculate_loss(x, beta=args.MAX_BETA, average=True,
                                                     eps=args.eps_gen)

            hist = {'val_loss': loss.item(), 'val_nll': -RE.item(),
                    'val_kl': KL.item(), 'val_reg': reg}
            history_elbo = {k: history_elbo[k] + hist[k] / N_steps for k in
                            set(history_elbo)}

        if len(args.input_size) > 2:
            model.visualize_results(args, args.dir, epoch, x[0:9])
    return history_elbo
