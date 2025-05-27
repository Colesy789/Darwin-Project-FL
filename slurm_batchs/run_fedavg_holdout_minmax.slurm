#!/bin/bash
#SBATCH --job-name=run_fedavg_holdout_minmax
#SBATCH --output=run_fedavg_holdout_minmax_%j.out
#SBATCH --error=run_fedavg_holdout_minmax_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Use a separate venv for this experiment
rm -rf ~/envs/flare_env_minmax

# Purge and load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0

python -m venv ~/envs/flare_env_minmax
source ~/envs/flare_env_minmax/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install nvflare==2.5.1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r /users/aca20whc/darwin/Darwin-Project-FL/flare_requirements.txt

# Confirm NVFlare installation
which nvflare
pip show nvflare

export PYTHONPATH=/users/aca20whc/darwin/Darwin-Project-FL/prostate_2D

# Move to new minmax holdout working directory
cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_minmax || {
  echo "Directory not found! Exiting."
  exit 1
}

# Clean old workspace for this run
rm -rf workspace_picai_minmax

# Run NVFlare simulator, set number of sites (-n) as appropriate for your data/splits/GPU config
nvflare simulator . -w ./workspace_picai_fedsemi_minmax -n 4 -t 4 -gpu 0,1

cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/

# # Example: run inference (uncomment & adjust as needed)
# python inference_seg_fl.py \
#   --workspace /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_holdout_minmax/workspace_picai_fedsemi_holdout_minmax \
#   --test_dir /mnt/parscratch/users/aca20whc/output/nnUNet_test_data \
#   --output_dir /users/aca20whc/darwin/Darwin-Project-FL/segmentation_result_minmax
