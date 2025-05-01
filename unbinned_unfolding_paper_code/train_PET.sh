#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes 32
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH -t 7:00:00
#SBATCH -A dasrepo
#SBATCH --gpu-bind=none
#SBATCH --module=gpu,nccl-2.18
#SBATCH -D /pscratch/sd/a/aelabd/omnifold_examples/ryans_way/unbinned_unfolding/unbinned_unfolding_paper_code
#SBATCH --mail-user=aelabd2@uw.edu
#SBATCH --mail-type=ALL

export TF_CPP_MIN_LOG_LEVEL=2

module load tensorflow
srun python train_PET.py --model-name "batch"