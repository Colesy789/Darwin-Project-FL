import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

def create_split_holdout(
    data_path, num_clients, test_fraction=0.10, output_path="split_holdout.json", seed=42
):
    random.seed(seed)
    data_path = Path(data_path)
    files = sorted([f.name for f in data_path.iterdir() if f.is_file()])
    if not files:
        raise ValueError(f"No files found in directory: {data_path}")

    # Shuffle before split
    random.shuffle(files)

    # Assign to clients round-robin
    client_file_map = defaultdict(list)
    for idx, file in enumerate(files):
        client_id = f"site-{(idx % num_clients) + 1}"
        client_file_map[client_id].append(file)
    
    # Now for each client, split into trainval and test
    client_splits = {}
    for client, client_files in client_file_map.items():
        n_files = len(client_files)
        n_test = max(1, int(test_fraction * n_files))
        random.shuffle(client_files)
        test_files = client_files[:n_test]
        trainval_files = client_files[n_test:]
        client_splits[client] = {
            "trainval": trainval_files,
            "test": test_files
        }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(client_splits, f, indent=2)

    print(
        f"Created split for {num_clients} clients (trainval+test per client), saved to {output_path}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate holdout split for FL with test per client.")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_path", type=str, default="split_holdout.json", help="Output path for JSON split")
    parser.add_argument("--test_fraction", type=float, default=0.10, help="Test split fraction per client (default 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    create_split_holdout(
        args.data_path,
        args.num_clients,
        output_path=args.output_path,
        test_fraction=args.test_fraction,
        seed=args.seed
    )
