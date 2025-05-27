#!/bin/bash
#SBATCH --job-name=run_fedavg_holdout_agnostic
#SBATCH --output=run_fedavg_holdout_agnostic_%j.out
#SBATCH --error=run_fedavg_holdout_agnostic_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Prepare dedicated Python environment for agnostic run
rm -rf ~/envs/flare_env_agnostic

module purge
module load Python/3.10.8-GCCcore-12.2.0

python -m venv ~/envs/flare_env_agnostic
source ~/envs/flare_env_agnostic/bin/activate

pip install --upgrade pip setuptools wheel
pip install nvflare==2.5.1
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r /users/aca20whc/darwin/Darwin-Project-FL/flare_requirements.txt

which nvflare
pip show nvflare

export PYTHONPATH=/users/aca20whc/darwin/Darwin-Project-FL/prostate_2D

# Change directory to your agnostic holdout config location
cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_agnostic || {
  echo "Directory not found! Exiting."
  exit 1
}

# Remove old workspace specific to this run
rm -rf workspace_picai_fedsemi_agnostic

# Run the NVFlare simulator for the agnostic FL holdout experiment
nvflare simulator . -w ./workspace_picai_fedsemi_agnostic -n 4 -t 4 -gpu 0,1

cd /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/

# # Uncomment and adjust for final inference step if needed
# python inference_seg_fl.py \
#   --workspace /users/aca20whc/darwin/Darwin-Project-FL/prostate_2D/job_configs/picai_fedsemi_holdout_agnostic/workspace_picai_fedsemi_holdout_agnostic \
#   --test_dir /mnt/parscratch/users/aca20whc/output/nnUNet_test_data \
#   --output_dir /users/aca20whc/darwin/Darwin-Project-FL/segmentation_result_agnostic
