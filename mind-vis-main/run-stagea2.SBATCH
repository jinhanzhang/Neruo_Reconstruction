#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/dr3631/my_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/dr3631/neuro/mind-vis-main;\ 
		python code/stageA2_mbm_finetune.py --dataset NOD \
		--pretrain_mbm_path results/fmri_pretrain/27-04-2024-17-20-24/checkpoints/checkpoint.pth"
