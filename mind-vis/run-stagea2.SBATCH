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
	    --overlay /scratch/jz5952/neuro-recon-stimuli-env/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "source /ext3/env.sh && cd /scratch/jz5952/mind-vis; python code/stageA2_mbm_finetune.py --dataset GOD --pretrain_mbm_path results/fmri_pretrain/07-05-2024-16-13-38/checkpoints/checkpoint.pth"
