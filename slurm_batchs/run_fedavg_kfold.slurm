#!/bin/bash
#SBATCH --job-name=run_fedavg_kfold
#SBATCH --output=run_fedavg_kfold_%j.out
#SBATCH --error=run_fedavg_kfold_%j.err
#SBATCH --time=10:00:00              # Adjust time as needed
#SBATCH --partition=gpu              # Your cluster's GPU partition
#SBATCH --gres=gpu:2                 # Request # GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Use a unique venv per run if needed, or share with other jobs if guaranteed to be thread-safe
rm -rf ~/envs/flare_env_kfold

module purge
module load Python/3.10.8-GCCcore-12.2.0

python -m venv ~/envs/flare_env_kfold
source ~/envs/flare_env_kfold/bin/activate

pip install --upgrade pip setuptools wheel
pip install nvflare==2.5.1

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r /users/aca20whc/darwin/Darwin-Project-FL/flare_requirements.txt

# Confirm NVFlare is installed (for debugging)
which nvflare
pip show nvflare

export PYTHONPATH=/users/aca20whc/darwin/Darwin-Project-FL/prostate_2D

cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_kfold || {
  echo "KFOLD directory not found! Exiting."
  exit 1
}

# Don't overwrite! Use unique kfold workspace
rm -rf workspace_picai_fedsemi_kfold

# Run NVFlare simulator for kfold
nvflare simulator . -w ./workspace_picai_fedsemi_kfold -n 4 -t 4 -gpu 0,1

cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/

# Optionally, add kfold-specific inference if needed

#python inference_seg_fl_kfold.py \
#  --workspace /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_kfold/workspace_picai_fedsemi_kfold
#  --test_dir /mnt/parscratch/users/aca20whc/output/nnUNet_test_data_kfold
#  --output_dir /users/aca20whc/darwin/Darwin-Project-FL/segmentation_result_kfold
