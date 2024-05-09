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
	    --overlay /scratch/jz5952/neuro-recon-stimuli-env/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "source /ext3/env.sh && cd /scratch/jz5952/mind-vis-main; python code/stageA2_mbm_finetune.py --dataset NOD --pretrain_mbm_path results/fmri_pretrain/27-04-2024-17-20-24/checkpoints/checkpoint.pth"
