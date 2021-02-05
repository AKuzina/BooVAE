from .continual import *
from .mnists import load_notmnist, load_mnist, load_fashion_mnist
from .celeba import load_celeba


def load_dataset(args, **kwargs):
    train_loader, val_loader, test_loader, args = {
        'mnist': load_mnist,
        'notmnist': load_notmnist,
        'fashion_mnist': load_fashion_mnist,
        'celeba': load_celeba
    }[args.dataset_name](args, **kwargs)

    return train_loader, val_loader, test_loader, args
