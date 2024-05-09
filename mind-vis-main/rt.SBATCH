#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu
#SBATCH --mem=128GB
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/dr3631/my_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/dr3631/neuro/mind-vis-main; python code/stageB_ldm_finetune.py --dataset NOD --pretrain_mbm_path results/fmri_finetune/28-04-2024-01-17-55/checkpoints/checkpoint.pth"
