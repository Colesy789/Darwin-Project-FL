#!/bin/bash
#SBATCH --job-name=detect_holdout
#SBATCH --output=detect_holdout_%j.out
#SBATCH --error=detect_holdout_%j.err
#SBATCH --time=95:00:00               # Adjust time as needed
#SBATCH --partition=gpu              # Replace with your cluster's GPU partition name
#SBATCH --gres=gpu:1                 # Request # GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

rm -rf ~/envs/flare_env_detect_holdout

# Purge and load modules
module purge

module load Python/3.10.8-GCCcore-12.2.0

python -m venv ~/envs/flare_env_detect_holdout

source ~/envs/flare_env_detect_holdout/bin/activate

# Upgrade pip and install correct versions of torch + torchvision
echo "Installing pip and nvflare"
pip install --upgrade pip setuptools wheel
pip install nvflare==2.5.1

echo "Installing tourch"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing requirements"
pip install -r /users/aca20whc/darwin/Darwin-Project-FL/flare_requirements.txt

# Confirm NVFlare is installed (for debugging)
#which nvflare
#pip show nvflare

export PYTHONPATH=/users/aca20whc/darwin/Darwin-Project-FL

# Move to working directory
echo "Finding Darwin-Project Directory"
cd /users/aca20whc/darwin/Darwin-Project-FL || {
  echo "Directory not found! Exiting."
  exit 1
}

echo "Running segmentation"
python segmentation/run_holdout.py -m train-cross
