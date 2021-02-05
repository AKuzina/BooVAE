import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets.continual import CustomDataset, select_subsample_boo


def load_celeba(args, **kwargs):
    # set args
    with args.unlocked():
        args.input_size = [3, 32, 32]
        args.input_type = 'continuous'
        args.min_inp = -1

    # preparing data
    assert args.max_tasks < 5
    if not args.incremental:
        args.max_tasks = 4
    hair_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'][:args.max_tasks]
    # 'Bald',

    img_transforms = transforms.Compose([
        transforms.Resize(45),
        transforms.CenterCrop((32, 32)),
        transforms.ToTensor()
    ])

    def subset_dset(dset):
        bad_attr = ['Wearing_Hat', 'Bald', 'Receding_Hairline']
        attr_ids = [np.where(np.array(dset.attr_names) == a)[0][0] for a in hair_attrs]
        bad_attr_ids = [np.where(np.array(dset.attr_names) == a)[0][0] for a in bad_attr]

        ## images with a chosen attribute (only one per attribute to avoid ambiguity)
        proper_ids = np.where((dset.attr[:, attr_ids].sum(1) == 1) &
                              (dset.attr[:, bad_attr_ids].sum(1) == 0))[0]
        ## All images, which have hair attributes (labels 0 -- max_task)
        dset.identity = np.argmax(dset.attr[:, attr_ids][proper_ids, :],
                                  axis=1).unsqueeze(1)
        dset.filename = dset.filename[proper_ids]
        dset.attr = dset.attr[proper_ids]
        loader = DataLoader(dset, batch_size=1000000)
        assert len(loader) == 1
        X, y = iter(loader).next()
        return X, y

    train = datasets.CelebA(root='datasets', download=True, split='train',
                            target_type='identity', transform=img_transforms)
    test = datasets.CelebA(root='datasets', download=True, split='test',
                           target_type='identity', transform=img_transforms)
    val = datasets.CelebA(root='datasets', download=True, split='valid',
                          target_type='identity', transform=img_transforms)

    x_train, y_train = subset_dset(train)
    x_test, y_test = subset_dset(test)
    x_val, y_val = subset_dset(val)

    print(x_train.shape)
    with args.unlocked():
        if args.incremental:
            args.all_tasks = np.array([[0], [1], [2], [3]])[:args.max_tasks]
        else:
            args.all_tasks = np.array([1])

    img_transforms_tr = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ColorJitter(brightness=0.1,
                                                                   contrast=0.1,
                                                                   saturation=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5))
                                            ])
    img_transforms_te = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5))
                                            ])
    train_dset = CustomDataset(X=x_train, y=y_train, transforms=img_transforms_tr,
                               all_tasks=args.all_tasks)
    test_dset = CustomDataset(X=x_test, y=y_test, transforms=img_transforms_te,
                              all_tasks=args.all_tasks)
    val_dset = CustomDataset(X=x_val, y=y_val, transforms=img_transforms_te,
                             all_tasks=args.all_tasks)

    with args.unlocked():
        if 'boost' in args.prior:
            args.X = select_subsample_boo(args, train_dset)
        else:
            args.X = [None]


    # pytorch data loader
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=4, **kwargs)
    val_loader = DataLoader(val_dset, batch_size=args.test_batch_size,
                                       shuffle=False, num_workers=4, **kwargs)
    test_loader = DataLoader(test_dset, batch_size=args.test_batch_size,
                                        shuffle=False, num_workers=4, **kwargs)

    # setting pseudo-inputs inits
    with args.unlocked():
        args.pseudoinputs_mean = 0.
        args.pseudoinputs_std = 0.3

    return train_loader, val_loader, test_loader, args