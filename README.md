
# Federated Learning Pipeline for Prostate Cancer Detection (PiCAI)

This repository contains the code to train a federated learning model using [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) for prostate cancer detection. The setup follows a semi-supervised federated learning scheme over 5 simulated clients. Then how to produce the results used for holdout, kfold, minmax, and agnostic testing within the report.

# Darwin Project FL - Setup and Preprocessing

This guide provides step-by-step instructions to set up and preprocess the Darwin Project FL on the University of Sheffield HPC. Please ensure you check any rootes in the files you are running and make sure they match your own personal rootes.

---

## 1. Connect to the VPN and HPC

```bash
# Connect to FortiClient VPN
# Then, open a terminal and connect to the HPC
ssh USERNAME@stanage.shef.ac.uk
```

---

## 2. Setup Project Directory

```bash
mkdir fl_darwin
cd fl_darwin
```

---

## 3. Create and Activate Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Clone the GitHub Repository

```bash
git clone https://github.com/Colesy789/Darwin-Project-FL
cd Darwin-Project-FL
```

---

## 5. Load Python Module and Set PYTHONPATH

```bash
pip install --upgrade pip
module load Python/3.10.8-GCCcore-12.2.0
export PYTHONPATH=/users/$USER/fl_darwin/Darwin-Project-FL
```

---

## 6. Install Requirements

### CUDA Dependencies

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Other Requirements

```bash
pip install -r flare_requirements.txt
```

---

## 7. Download and Prepare Picai Data

```bash
mkdir /mnt/parscratch/users/$USER
chmod 700 /mnt/parscratch/users/$USER
cd /mnt/parscratch/users/$USER

mkdir picai_combined
cd picai_combined
wget https://zenodo.org/records/6624726/files/picai_public_images_fold{0..4}.zip?download=1

for f in *.zip*; do unzip "$f" -d . && rm "$f"; done

cd /mnt/parscratch/users/$USER
mkdir picai_combined_labels
```

- Download labelled data from: [https://github.com/DIAGNijmegen/picai_labels/tree/main/csPCa_lesion_delineations/human_expert/resampled](https://github.com/DIAGNijmegen/picai_labels/tree/main/csPCa_lesion_delineations/human_expert/resampled)
- Use [https://downgit.github.io/#/home](https://downgit.github.io/#/home) to download it.

```bash
scp "local_path_to_downloaded_file" USERNAME@stanage.shef.ac.uk:/mnt/parscratch/users/USERNAME/
unzip resampled.zip -d /mnt/parscratch/users/$USER/picai_combined_labels
rm /mnt/parscratch/users/$USER/resampled.zip

mkdir work output splits
```

---

## 8. Run Preprocessing

```bash
cd ~/fl_darwin/Darwin-Project-FL
python preprocess/preprocess.py --workdir=/mnt/parscratch/users/$USER/work --imagesdir=/mnt/parscratch/users/$USER/picai_combined --labelsdir=/mnt/parscratch/users/$USER/picai_combined_labels --outputdir=/mnt/parscratch/users/$USER/output --splits=/mnt/parscratch/users/$USER/splits
```

---

## 9. Modify Classification Paths

- Edit `classification/cls_data.py`:
  - Lines 107-109, 156-158: update paths with `/mnt/parscratch/users/USERNAME/` and `/users/USERNAME/`

- Run data generation:
```bash
python classification/cls_data.py -m make_data
```

- Edit `classification/config.py` Line 18 to:
```python
CSV_PATH = '/mnt/parscratch/users/USERNAME/output/classification/picai_illness_3c.csv'
```

---

## 10. Train and Predict with SLURM

```bash
cd ~/fl_darwin/slurm_batches

# Training
sbatch run_training_phase.slurm
squeue -u $USER

# Prediction
sbatch run_prediction_phase.slurm
squeue -u $USER
```

---

## 11. Federated Training

### Holdout

```bash
cd ~/fl_darwin/Darwin-Project-FL
python generate_split_holdout.py --num_clients 4 --data_path /mnt/parscratch/users/$USER/output/segmentation/segdata/data_2d

cd ~/fl_darwin/slurm_batches
sbatch run_fedavg_holdout_cs.slurm
```

### K-Fold

```bash
python generate_split_kfold.py --num_clients 4 --num_folds 5 --data_path /mnt/parscratch/users/$USER/output/segmentation/segdata/data_2d --output_path split_kfold.json

sbatch run_fedavg_kfold.slurm
```

### Minmax Holdout

```bash
sbatch run_fedavg_holdout_minmax.slurm
```

### Agnostic Holdout

```bash
sbatch run_fedavg_holdout_agnostic.slurm
```

---

## 12. Inference

```bash
sbatch inf_holdout_cv.slurm
sbatch inf_kfold.slurm
sbatch inf_holdout_minmax.slurm
sbatch inf_holddout_agnostic.slurm
```

---

## 13. Dataset Preparation for Detection

- Edit lines 185-190 in `segmentation/make_dataset.py` for each inference result
- Example for holdout:
```python
base_dir = '/mnt/parscratch/users/USERNAME/output/nnUNet_raw_data/Task2201_picai_baseline/imagesTr'
label_dir = '/mnt/parscratch/users/USERNAME/output/nnUNet_raw_data/Task2201_picai_baseline/labelsTr'
output_dir = '/mnt/parscratch/users/USERNAME/output/segmentation/detectdata/holdout_cv'
test_dir = '/mnt/parscratch/users/USERNAME/output/nnUNet_test_data'
seg_dir = '/users/USERNAME/fl_darwin/Darwin-Project-FL/seg_result_holdout_cv'
csv_path = '/users/USERNAME/fl_darwin/Darwin-Project-FL/classification_result/test_3c.csv'
```

```bash
python make_data
```

---

## 14. Detection

```bash
sbatch detect_holdout.slurm
sbatch detect_kfold.slurm
sbatch detect_minmax.slurm
sbatch detect_agnostic.slurm
```

---

## 15. Calculate Results

- Edit `calc_results.py` Line 6:
```python
ROOT_DIR = "/users/USERNAME/fl_darwin/Darwin-Project-FL/new_ckpt/detect/itunet_d24_holdout"
```

Run after each change:
```bash
python calc_results.py
```

---

**Results will be printed in the terminal.**
