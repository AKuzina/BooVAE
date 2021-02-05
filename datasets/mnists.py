import os
import pickle

import PIL
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from datasets.continual import CustomDataset, select_subsample_boo

ALL_TASKS = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])


def train_val_prep(train_dset):
    x_train = train_dset.data.float().numpy() / 255.
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

    y_train = np.array(train_dset.targets.float().numpy(), dtype=int)

    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)
    return x_train, y_train, x_val, y_val


def load_notmnist(args, **kwars):
    # set args
    with args.unlocked():
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        args.min_inp = 0

    link_to_data = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat'
    path_to_data = 'datasets/notMNIST'
    if not os.path.exists(path_to_data):
        datasets.utils.download_url(link_to_data, root=path_to_data,
                                    filename='notMNIST_small.mat')
    out = loadmat(os.path.join(path_to_data, 'notMNIST_small.mat'))
    X, Y = out['images'], out['labels']
    Y = np.array(Y, dtype=int)
    X = X.transpose(2, 0, 1)
    X = X.reshape(X.shape[0], -1)
    X = np.array(X, dtype=np.float32)
    X /= 255.0

    seed = 0
    np.random.seed(seed)
    N_train = int(X.shape[0] * 0.9)
    ind = np.random.permutation(range(X.shape[0]))
    x_train = X[ind[:N_train]]
    y_train = Y[ind[:N_train]]
    x_test = X[ind[N_train:]]
    y_test = Y[ind[N_train:]]

    N_train = int(x_train.shape[0] * 0.9)
    x_valid = x_train[N_train:]
    y_valid = y_train[N_train:]
    x_train = x_train[:N_train]
    y_train = y_train[:N_train]

    with args.unlocked():
        if args.incremental:
            args.all_tasks = ALL_TASKS[:args.max_tasks]
        else:
            args.all_tasks = np.array([1])

    img_transforms = None
    train_dset = CustomDataset(X=torch.from_numpy(x_train),
                               y=torch.from_numpy(y_train),
                               transforms=img_transforms)
    test_dset = CustomDataset(X=torch.from_numpy(x_test), y=torch.from_numpy(y_test),
                              transforms=img_transforms)
    val_dset = CustomDataset(X=torch.from_numpy(x_valid),
                             y=torch.from_numpy(y_valid),
                             transforms=img_transforms)

    with args.unlocked():
        if 'boost' in args.prior:
            args.X = select_subsample_boo(args, train_dset)
        else:
            args.X = [None]

    # pytorch data loader
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=args.test_batch_size,
                            shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=0)

    # setting pseudo-inputs inits
    with args.unlocked():
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.2

    return train_loader, val_loader, test_loader, args


def load_mnist(args, **kwargs):
    # set args
    with args.unlocked():
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        args.min_inp = 0

    train_dset = datasets.MNIST('datasets', train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))

    test_dset = datasets.MNIST('datasets', train=False,
                               transform=transforms.Compose([transforms.ToTensor()]))

    x_train, y_train, x_val, y_val = train_val_prep(train_dset)
    x_test = test_dset.data.float().numpy() / 255.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    y_test = np.array(test_dset.targets.float().numpy(), dtype=int)

    with args.unlocked():
        if args.incremental:
            args.all_tasks = ALL_TASKS[:args.max_tasks]
        else:
            args.all_tasks = np.array([1])

    # pytorch data loader
    img_transforms = None
    train_dset = CustomDataset(X=torch.from_numpy(x_train).float(),
                               y=torch.from_numpy(y_train),
                               transforms=img_transforms, all_tasks=args.all_tasks)
    test_dset = CustomDataset(X=torch.from_numpy(x_test).float(),
                              y=torch.from_numpy(y_test),
                              transforms=img_transforms, all_tasks=args.all_tasks)
    val_dset = CustomDataset(X=torch.from_numpy(x_val).float(),
                             y=torch.from_numpy(y_val),
                             transforms=img_transforms, all_tasks=args.all_tasks)
    with args.unlocked():
        if 'boost' in args.prior:
            args.X = select_subsample_boo(args, train_dset)
        else:
            args.X = [None]

    # pytorch data loader
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=args.test_batch_size,
                            shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=0)
    with args.unlocked():
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.15

    return train_loader, val_loader, test_loader, args


def load_fashion_mnist(args, **kwargs):
    # set args
    with args.unlocked():
        args.input_size = [1, 28, 28]
        args.input_type = 'binary'
        args.min_inp = 0

    train_dset = datasets.FashionMNIST('datasets', train=True, download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]))

    test_dset = datasets.FashionMNIST('datasets', train=False,
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]))

    x_train, y_train, x_val, y_val = train_val_prep(train_dset)

    x_test = test_dset.data.float().numpy() / 255.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    y_test = np.array(test_dset.targets.float().numpy(), dtype=int)

    with args.unlocked():
        if args.incremental:
            args.all_tasks = ALL_TASKS[:args.max_tasks]
        else:
            args.all_tasks = np.array([1])

    # pytorch data loader
    img_transforms = None
    train_dset = CustomDataset(X=torch.from_numpy(x_train).float(),
                               y=torch.from_numpy(y_train),
                               transforms=img_transforms)
    test_dset = CustomDataset(X=torch.from_numpy(x_test).float(),
                              y=torch.from_numpy(y_test),
                              transforms=img_transforms)
    val_dset = CustomDataset(X=torch.from_numpy(x_val).float(),
                             y=torch.from_numpy(y_val),
                             transforms=img_transforms)
    with args.unlocked():
        if 'boost' in args.prior:
            args.X = select_subsample_boo(args, train_dset)
        else:
            args.X = [None]

    # pytorch data loader
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=args.test_batch_size,
                            shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=0)
    with args.unlocked():
        args.pseudoinputs_mean = 0.5
        args.pseudoinputs_std = 0.15

    return train_loader, val_loader, test_loader, args