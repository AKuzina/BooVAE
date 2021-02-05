import os

def get_model_name(args):
    model_name = ''
    if args.multihead:
        model_name += 'multi'
    if args.regularizer:
        model_name += '{}'.format(args.reg_type, args.reg_weight)
        if args.reg_type in ['ewc', 'si', 'vcl']:
            model_name += '{}'.format(args.reg_weight)
        model_name += '_'

    model_name += args.prior
    if args.prior != 'standard':
         model_name += '(K' + str(args.number_components)
    model_name += '_wu' + str(args.warmup)
    if args.wd > 0:
        model_name += '_{}wd'.format(args.wd)
    if args.coreset_size > 0:
        model_name += '_{}core'.format(args.coreset_size)
    if args.self_replay > 0:
        model_name += '_{}replay'.format(args.self_replay)

    model_name += ')_z1_{}'.format(str(args.z1_size))
    if args.multihead:
        model_name += '_z2_{}'.format(str(args.z2_size))
    # CREATE MODEL NAME

    if args.input_type != 'binary':
        model_name += args.ll

    if 'boost' in args.prior:
        model_name += '_({}E_{}lbd_{}epsgen'.format(
            # int(args.boost_steps / 1000),
            args.comp_ep,
            int(args.lbd),
            args.eps_gen)

        if args.component_weight == 'fixed':
            model_name += '_fixed)'
        elif args.component_weight == 'grad':
            model_name += '_grad)'
        else:
            model_name += ')'
    elif 'incr' in args.prior:
        model_name += '_({}epsgen)'.format(args.eps_gen)
    return model_name


def model_path(args, model_name):
    dir = 'runs/'
    if args.incremental:
        dir = os.path.join(dir, 'incremental', args.dataset_name, model_name,
                           'task_'+str(args.max_tasks), 'iter_'+str(args.iter))
    else:
        dir = os.path.join(dir, 'simple', args.dataset_name, model_name, 'iter_'+str(args.iter))

    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
