import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold  # You’ll need scikit-learn installed

def create_kfold_split(data_path, num_clients, num_folds=5, output_path="split_kfold.json", seed=42):
    random.seed(seed)
    data_path = Path(data_path)

    # Get list of file names in data_path
    files = sorted([f.name for f in data_path.iterdir() if f.is_file()])
    if not files:
        raise ValueError(f"No files found in directory: {data_path}")

    # Shuffle files
    random.shuffle(files)
    # Split files across clients as before
    client_files = defaultdict(list)
    for idx, file in enumerate(files):
        client_id = f"site-{(idx % num_clients) + 1}"
        client_files[client_id].append(file)

    # Now, for each client, split into K-folds for local validation
    client_splits = {}
    for client_id, files in client_files.items():
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        client_splits[client_id] = {}
        file_arr = list(files)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(file_arr)):
            train_files = [file_arr[i] for i in train_idx]
            val_files = [file_arr[i] for i in val_idx]
            client_splits[client_id][f"fold_{fold_idx}"] = {
                "train": train_files,
                "val": val_files
            }

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(client_splits, f, indent=2)

    print(f"Data split for {num_clients} clients, {num_folds} folds each, saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data among clients and make k-fold splits")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for k-fold (default: 5)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_path", type=str, default="split_kfold.json", help="Output JSON path")
    args = parser.parse_args()

    create_kfold_split(
        args.data_path,
        args.num_clients,
        args.num_folds,
        args.output_path
    )
