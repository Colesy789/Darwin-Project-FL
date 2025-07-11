#!/bin/bash
#SBATCH --job-name=inf_holdout_minmax
#SBATCH --output=inf_holdout_minmax_%j.out
#SBATCH --error=inf_holdout_minmax_%j.err
#SBATCH --time=10:00:00              # Adjust time as needed
#SBATCH --partition=gpu              # Replace with your cluster's GPU partition name
#SBATCH --gres=gpu:1                 # Request # GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

rm -rf ~/envs/flare_env_minmax

# Purge and load modules
module purge

module load Python/3.10.8-GCCcore-12.2.0

python -m venv ~/envs/flare_env_minmax

source ~/envs/flare_env_minmax/bin/activate

# Upgrade pip and install correct versions of torch + torchvision
pip install --upgrade pip setuptools wheel
pip install nvflare==2.5.1

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r /users/aca20whc/darwin/Darwin-Project-FL/flare_requirements.txt

# Confirm NVFlare is installed (for debugging)
which nvflare
pip show nvflare

export PYTHONPATH=/users/aca20whc/darwin/Darwin-Project-FL/prostate_2D

cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/

# run inference
python run_inference.py \
  --workspace /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_minmax/workspace_picai_fedsemi_minmax \
  --test_dir /mnt/parscratch/users/aca20whc/output/nnUNet_test_data \
  --output_dir /users/aca20whc/darwin/Darwin-Project-FL/seg_result_holdout_minmax
