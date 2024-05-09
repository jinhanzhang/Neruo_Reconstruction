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
		python code/gen_eval.py --dataset NOD \
		--model_path results/generation3/07-05-2024-22-03-25/checkpoint_best.pth \
		--pretrain_mbm_path results/fmri_finetune/07-05-2024-20-31-32/checkpoints/checkpoint.pth"
