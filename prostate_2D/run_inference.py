import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
from custom.utils.training_utils import SemanticSeg
from tqdm import tqdm

def load_model(model_path, device):
    model = SemanticSeg(
        lr=1e-4,
        n_epoch=1,
        channels=3,
        num_classes=2,
        input_shape=(384, 384),
        batch_size=1,
        num_workers=2,
        device="0",
        pre_trained=False,
        ckpt_point=False,
        use_fp16=False,
        transformer_depth=18,
        use_transfer_learning=True
    ).net

    checkpoint = torch.load(model_path, map_location=device)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return model

def run_inference(model, test_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(sorted(os.listdir(test_dir))):
        if not file.endswith("_0000.nii.gz"):
            continue

        subject_id = file.replace("_0000.nii.gz", "")
        paths = [os.path.join(test_dir, f"{subject_id}_{i:04d}.nii.gz") for i in range(3)]

        if not all(os.path.exists(p) for p in paths):
            print(f"[WARN] Missing modalities for {subject_id}, skipping.")
            continue

        imgs = [sitk.ReadImage(p) for p in paths]
        arrays = [sitk.GetArrayFromImage(im) for im in imgs]
        image_np = np.stack(arrays)  # [3, D, H, W]

        pred_mask = np.zeros(image_np.shape[1:], dtype=np.uint8)  # [D, H, W]
        for i in range(image_np.shape[1]):
            input_slice = image_np[:, i, :, :]
            input_tensor = torch.tensor(input_slice.astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            pred_mask[i] = pred

        output_path = os.path.join(output_dir, f"{subject_id}.npy")
        np.save(output_path, pred_mask)
        print(f"[INFO] Saved prediction: {output_path}")

def run_inference_softmax(model, test_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(sorted(os.listdir(test_dir))):
        if not file.endswith("_0000.nii.gz"):
            continue

        subject_id = file.replace("_0000.nii.gz", "")
        paths = [os.path.join(test_dir, f"{subject_id}_{i:04d}.nii.gz") for i in range(3)]

        if not all(os.path.exists(p) for p in paths):
            print(f"[WARN] Missing modalities for {subject_id}, skipping.")
            continue

        imgs = [sitk.ReadImage(p) for p in paths]
        arrays = [sitk.GetArrayFromImage(im).astype(np.float32) for im in imgs]
        image_np = np.stack(arrays)  # [3, D, H, W]

        C, D, H, W = 2, image_np.shape[1], image_np.shape[2], image_np.shape[3]
        volume = np.zeros((C, D, H, W), dtype=np.float32)

        for i in range(D):
            input_slice = image_np[:, i, :, :].copy()  # [3, H, W]

            # Normalize each modality (channel) individually
            for c in range(input_slice.shape[0]):
                max_val = np.max(input_slice[c])
                if max_val != 0:
                    input_slice[c] = input_slice[c] / max_val

            input_tensor = torch.tensor(input_slice).unsqueeze(0).to(device)  # [1, 3, H, W]

            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                softmax = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()  # [C, H, W]

            volume[:, i, :, :] = softmax

        output_path = os.path.join(output_dir, f"{subject_id}.npy")
        np.save(output_path, volume)
        print(f"[INFO] Saved softmax prediction: {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Run inference using FL-trained prostate segmentation model.")
    parser.add_argument("--workspace", type=str, required=True, help="Base workspace directory path.")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test .nii.gz files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predicted masks.")
    args = parser.parse_args()

    # Auto-construct model path from workspace
    model_path = os.path.join(args.workspace, "server/simulate_job/app_server/FL_global_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    print("Model loaded. Starting inference...")
    run_inference_softmax(model, args.test_dir, args.output_dir, device)

if __name__ == "__main__":
    main()