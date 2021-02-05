# BooVAE: Boosting Approach for Continual Learning of VAEs

---

PyTorch implementation of the paper:
 
 [BooVAE: Boosting Approach for Continual Learning of VAEs]()\\
 [Anna Kuzina](https://akuzina.github.io/)\*, [Evgenii Egorov](http://evgenii-egorov.github.io)\*, [Evgeny Burnaev](https://faculty.skoltech.ru/people/evgenyburnaev)
 
 \* - equal contribution

## Abstract
Variational autoencoder (VAE) is a deep generative model for unsupervised learning, allowing to encode observations into the meaningful latent space. VAE is prone to catastrophic forgetting when tasks arrive sequentially, and only the data for the current one is available. We address this problem of continual learning for VAEs. It is known that the choice of the prior distribution over the latent space is crucial for VAE in the non-continual setting. We argue that it can also be helpful to avoid catastrophic forgetting. We learn the approximation of the aggregated posterior as a prior for each task. This approximation is parametrised as an additive mixture of distributions induced by encoder evaluated at trainable pseudo-inputs. We use a greedy boosting-like approach with entropy regularisation to learn the components. This method encourages components diversity, which is essential as we aim at memorising the current task with the fewest components possible. Based on the learnable prior, we introduce an end-to-end approach for continual learning of VAEs and provide empirical studies on commonly used benchmarks (MNIST, Fashion MNIST, NotMNIST) and CelebA datasets. For each dataset, the proposed method avoids catastrophic forgetting in a fully automatic way.


## Experiments
#### Environment setup
 
The exact specification of our environment is provided in the file `environment.yml` and
can be created via 
```bash
conda env create -f environment.yml
```

The command above will create as environment `boovae` with all the required dependencies. 

#### Experiments for the paper
```bash
python run_experiment.py ...
```

## Citation
If you find our paper or code useful, feel free to cite:
```text
@article{

} 
```


