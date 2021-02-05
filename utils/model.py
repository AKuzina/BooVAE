import os
import torch
import numpy as np


from vae.utils.architectures import get_architecture
from utils.naming import model_path


def get_model(args):
    
    if args.prior == 'standard':
        import vae.model.standard as models
    elif 'mog' in args.prior:
        import vae.model.mog as models
    elif args.prior == 'boost':
        import vae.model.boost as models
        
    arc, args = get_architecture(args)
    
    if args.input_type == 'continuous':
        inp = args.input_size
    else:
        inp = [np.prod(args.input_size)]

    with args.unlocked():
        args.min_task = 0
    if args.resume:
        print('Resume training model', args.model_name)
        model = torch.load(os.path.join(args.dir, 'model.pth'),
                           map_location=torch.device(args.device))
    elif args.incr_resume and args.max_tasks > 1:
        args.max_tasks -= 1
        path_to_prev_model = model_path(args, args.model_name)
        print('resuming from: ', path_to_prev_model)
        args.max_tasks += 1
        model = torch.load(os.path.join(path_to_prev_model, 'model.pth'),
                           map_location=torch.device(args.device))
        args.min_task = args.max_tasks - 1
        args.all_tasks = args.all_tasks[args.min_task:]
    else:
        model = models.Simple(args.z1_size, args.input_type, args.ll, arc,
                              incremental=args.incremental, input_size=inp,
                              # use_training_data_init=args.use_training_data_init,
                              pseudoinputs_mean=args.pseudoinputs_mean,
                              pseudoinputs_std=args.pseudoinputs_std,
                              number_components=args.number_components, X_opt=args.X[0],
                              comp_weight=args.component_weight, min_inp=args.min_inp,
                              scale=args.scale)

    model.to(args.device)
    return model, args