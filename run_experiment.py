import os
import sys
import wandb
import ml_collections
import copy

import torch
import numpy as np

from datasets import load_dataset
from utils.naming import get_model_name, model_path
from utils.model import get_model
from utils.trainer import train_vae, get_optimizer
from utils.tester import test_vae

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py")


def get_beta(args):
    with args.unlocked():
        args.MAX_BETA = {
            'mnist': 1,
            'fashion_mnist': 1,
            'notmnist': 1,
            'celeba': 2
        }[
            args.dataset_name
        ]
    return args


def run(_):
    if "absl.logging" in sys.modules:
        import absl.logging

        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")
    args = FLAGS.config
    print(args)
    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # set arguments for regularization
    with args.unlocked():
        args.bayes = False
        if args.reg_type in ['ewc', 'vcl', 'si']:
            args.regularizer = True
            if args.reg_type == 'vcl':
                args.bayes = True
        else:
            args.regularizer = False

    # setup max beta based on the dataset
    args = get_beta(args)

    # get dataloaders
    train_loader, val_loader, test_loader, args = load_dataset(args)

    # CREATE MODEL NAME
    with args.unlocked():
        args.model_name = get_model_name(args)
        args.dir = model_path(args, args.model_name)

    if args.self_replay > 0:
        args.max_tasks -= 1
        model_name = get_model_name(args)
        with args.unlocked():
            args.prev_model_path = os.path.join(model_path(args, model_name), 'model.pth')
        args.max_tasks += 1

    # initialize weight and bias
    os.environ["WANDB_API_KEY"] = # API KEY HERE
    tags = [
        args.prior,
        args.dataset_name
    ]
    if args.incremental:
        tags.append(str(args.max_tasks))

    wandb.init(
        project="boovae",
        config=copy.deepcopy(dict(args)),
        group=args.dataset_name,
        tags=tags,
    )

    # IMPORT AND INIT THE MODEL
    print('creating model {}'.format(args.model_name))
    model, args = get_model(args)

    # GET OPTIMIZER
    optimizer, scheduler, boost_optimizer = get_optimizer(args, model)
    # TRAIN
    train_vae(args, train_loader, val_loader, model, optimizer, scheduler, boost_optimizer)
    # TEST
    test_vae(args, test_loader)

if __name__ == "__main__":
    app.run(run)