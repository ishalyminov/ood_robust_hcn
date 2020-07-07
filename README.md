# OOD-robust Hybrid Code Networks
 
A suite of methods for improving robustness of task-oriented dialogue models.

Code for paper "Improving Robustness of Dialog Systems in a Data-Efficient Way with Turn Dropout" by [Igor Shalyminov](https://github.com/ishalyminov) and [Sungjin Lee](https://www.linkedin.com/in/sungjinlee/). [[Paper - presented at ConvAI Workshop@NeurIPS 2018]](https://arxiv.org/abs/1811.12148) [[Poster]](https://drive.google.com/file/d/1c6Kd3aGwaEj4tgyElIe1dYgAwM-B9aRd/view?usp=sharing)

HCN implementation is based on https://github.com/johndpope/hcn

[Repo with OOD-augmented data](https://github.com/sungjinl/icassp2019-ood-dataset)

Setup with Conda
==

1. `conda create -n ood_robust_hcn python=3.7 cython tensorflow-gpu==1.14.0`

2. `conda activate ood_robust_hcn`

3. `pip install -r requirements.txt`

Dialog control with HCN
==

0.1 Download word2vec vectors:

`cd hcn/data; sh get_word2vec.sh`

0.2 Initialize the datasets

`git submodule update --init`

`cd icassp-ood-dataset; unzip *.zip`

1. Training:

`cd hcn; python train.py data ../icassp-ood-dataset/babi_task6 ../icassp-ood-dataset/babi_task6_ood_0.2_0.4 <model folder> configs/<config-json> [--custom_vocab <vocab file>]`

2. Evaluation:

`cd hcn; python evaluate.py data ../icassp-ood-dataset/babi_task6 ../icassp-ood-dataset/babi_task6_ood_0.2_0.4 <model folder> [clean/noisy]`

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

`cd vae; python evaluate_vae_ood.py <model folder> <AE dataset folder>/devset <AE dataset folder>/evalset --decision_type [min/max/avg] --loss_components [kl_loss(,nll_loss)]``

Custom OOD data generation
==
bAbI Dialog Task 6 augmentation:

1. Run the notebooks:

`mining_ood_reddit.ipynb, mining_ood_twitter.ipynb, mining_foreign_domain_ood.ipynb, mining_ood_breakdown.ipynb`

2. `cd babi_tools; sh make_ood_dataset.sh ../hcn/data ../data/babi_task6_ood_dataset_<parameters>`

`ood_augmentation.json` config file will be used which sets the probabilities of OOD sequence start and continuation respectively.

