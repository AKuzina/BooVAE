import ml_collections


def get_config():
    default_config = dict(
    ### OPTIMIZATION
        # learning rate (initial if cheduler is used)
        lr=5e-4,
        # weight decay
        wd=0,
        early_stopping_epochs=50,
        # num epochs to anneal \beta
        warmup=100,

    ### VAE
        # latent size
        z1_size=40,
        # prior: standard, mog, boost
        prior='standard',
        # type of the loglikelihood for continuous input: logistic, normal, l1
        ll='logistic',

    ### BooVAE
        # How often to update component of the prior h (in epochs)
        comp_ep=1,
        # Prune boosting components
        prune=True,
        lbd=1,
        # Weight of generation regularizer
        eps_gen=0.,
        # ?????
        # number of epoch before start adding components
        # init_epochs=0,
        # max number of gradient steps to train boosting component
        # boost_steps=25000,
        # equal / fixed / grad/  tba
        component_weight='equal',
        # Real to pseudoinput rati
        scale=1,
        # number of pseudo-inputs in the prior
        number_components=500,

    ### CORESETS
        # number of random coreset samples to add to training dataset
        coreset_size=0,

    ### MULTIHEAD
        multihead=False,
        # shared latent size
        z2_size=40,
        
    ### WEIGHT REGULARIZATION
        reg_weight=0,
        # Regularizer type (ewc, si, vcl)
        reg_type=None,

    ### SELF-REPLAY
        self_replay=0,

    ### EXPERIMENT
        # iter number to evaluate std
        iter=0,
        # input batch size for training and testing
        batch_size=500,
        test_batch_size=500,
        epochs=100,
        device='cuda',
        seed=14,
        dataset_name='mnist',
        # number of samples used for approximating log-likelihood
        S=1000,
        # If load pretrained model
        resume=False,

    ### CONTINUAL LEARNING
        incremental=True,
        max_tasks=2,
        # continue training (start from the last task)
        incr_resume=True
    )

    default_config = ml_collections.ConfigDict(default_config)
    return default_config