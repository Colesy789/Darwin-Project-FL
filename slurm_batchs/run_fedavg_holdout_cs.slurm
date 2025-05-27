#!/bin/bash
#SBATCH --job-name=run_fedavg_holdout_cs
#SBATCH --output=run_fedavg_holdout_cs_%j.out
#SBATCH --error=run_fedavg_holdout_cs_%j.err
#SBATCH --time=10:00:00 # Adjust as needed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Remove any existing venv if you want a clean environment every time (optional)
rm -rf ~/envs/flare_env_holdout

# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0

# Create and activate a new virtual environment for holdout CS run
python -m venv ~/envs/flare_env_holdout
source ~/envs/flare_env_holdout/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install nvflare==2.5.1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r /users/aca20whc/darwin/Darwin-Project-FL/flare_requirements.txt

# Confirm NVFlare installation
which nvflare
pip show nvflare

export PYTHONPATH=/users/aca20whc/darwin/Darwin-Project-FL/prostate_2D

# Move to the new job config directory for the holdout run
cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_holdout || {
    echo "Directory not found! Exiting."
    exit 1
}

# Clean any old workspace for this run
rm -rf workspace_picai_fedsemi_holdout

# Run NVFlare simulator for the holdout setting
nvflare simulator . -w ./workspace_picai_fedsemi_holdout -n 4 -t 4 -gpu 0,1

cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/

# Example: run inference using NEW workspace/output dir (if needed)
# python inference_seg_fl.py \
#   --workspace /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_holdout/workspace_picai_fedsemi_holdout \
#   --test_dir /mnt/parscratch/users/aca20whc/output/nnUNet_test_data \
#   --output_dir /users/aca20whc/darwin/Darwin-Project-FL/segmentation_result_cs_holdout
