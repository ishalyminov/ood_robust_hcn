#!/bin/bash

#SBATCH --partition=amd-longq
#SBATCH --nodes 1
#SBATCH --mail-user=is33@hw.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --mem=256G

# do some stuff to setup the environment
CUDA_VERSION=cuda90
CUDA_VERSION_LONG=9.0.176
CUDNN_VERSION=7.0
module purge
module load shared
module load $CUDA_VERSION/blas/$CUDA_VERSION_LONG $CUDA_VERSION/fft/$CUDA_VERSION_LONG $CUDA_VERSION/nsight/$CUDA_VERSION_LONG $CUDA_VERSION/profiler/$CUDA_VERSION_LONG $CUDA_VERSION/toolkit/$CUDA_VERSION_LONG cudnn/$CUDNN_VERSION

# execute application (read in arguments from command line)
# e.g. learning_to_rank <trainset file> <devset file> <testset file> <model folder>
cd /home/ishalyminov/data/ood_robust/hcn && /home/ishalyminov/Envs/ood_robust/bin/python -m train $@

# exit
exit 0
