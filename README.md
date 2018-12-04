# OOD-robust Hybrid Code Networks
 
A suite of methods for improving robustness of task-oriented dialogue models.

Code for paper "Improving Robustness of Dialog Systems in a Data-Efficient Way with Turn Dropout" by [Igor Shalyminov](https://github.com/ishalyminov) and [Sungjin Lee](https://www.microsoft.com/en-us/research/people/sule/). [[Paper]](https://arxiv.org/abs/1811.12148) [[Poster]](https://drive.google.com/file/d/1c6Kd3aGwaEj4tgyElIe1dYgAwM-B9aRd/view?usp=sharing)

HCN implementation is based on https://github.com/johndpope/hcn

Data coming soon!

OOD data generation
==
bAbI Dialog Task 6 augmentation:

`cd babi_tools; sh make_ood_dataset.sh ../hcn/data ../data/babi_task6_ood_dataset_<parameters>`

`ood_augmentation.json` config file will be used which sets the probabilities of OOD sequence start and continuation respectively.


Standalone OOD detection
==
1. Autoencoder-based

Making a dataset for AE:

`python make_dataset_for_autoencoder.py hcn/data <result folder>`

Training an AE:

`cd ae_ood; python train_ae.py <AE dataset folder>/trainset <AE dataset folder>/devset <AE dataset folder>/testset <model folder>`

Evaluating the AE:

`cd ae_ood; python evaluate.py <model folder> <AE dataset folder>/devset <AE dataset folder>/evalset --decision_type [min/max/avg]`

2. VAE-based

Training a VAE:

`cd vae; python train.py <AE dataset folder>/trainset <AE dataset folder>/devset <AE dataset folder>/testset <model folder>`

Evaluating the VAE:

`cd vae; python evaluate_vae_ood.py <model folder> <AE dataset folder>/devset <AE dataset folder>/evalset --decision_type [min/max/avg] --loss_components [kl_loss(,nll_loss)]`

Dialog control with HCN
==

Training:

`cd hcn; python train.py data ../data/babi_task6_ood_dataset_<parameters> <model folder> config.json --custom_vocab <vocab file>`

Evaluation:

`cd hcn; python evaluate.py data ../data/babi_task6_ood_dataset_<parameters> <model folder> [clean/noisy]`
