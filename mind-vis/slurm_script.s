#!/bin/bash
#SBATCH --job-name=jzfov
#SBATCH --output=output_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=3:59:00
#SBATCH --gres=gpu:1

singularity exec --nv --overlay /scratch/jz5952/neuro-recon-stimuli-env/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c "source /ext3/env.sh && cd /scratch/jz5952/mind-vis && python code/stageA1_mbm_pretrain.py --num_epoch 200 --model_name MAEforFMRI"
# python code/stageB_ldm_finetune.py --dataset GOD --pretrain_mbm_path results/fmri_pretrain/27-04-2024-17-20-24/checkpoints/checkpoint.pth"
# python code/stageA2_mbm_finetune.py --dataset GOD --pretrain_mbm_path results/fmri_pretrain/27-04-2024-17-20-24/checkpoints/checkpoint.pth
# python code/stageA1_mbm_pretrain.py
# source /ext3/env.sh
# cd /scratch/jz5952/FoV

# # Run your Python code
# python3 main.py --model MyTransformer  --num_epochs 50 --hist_time 1.0 --pred_time 0.1
