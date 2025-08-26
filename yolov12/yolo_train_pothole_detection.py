#!/usr/bin/env python3
"""
YOLOv12 trainer (Apple Silicon / MPS) — clean Ultralytics style

Example:
  python3 train_pothole_yolo.py
"""
import torch
torch.backends.mps.allow_tf32 = False
torch.set_float32_matmul_precision("high")
from ultralytics import YOLO
from pathlib import Path
import os

DATA_YAML = (Path(__file__).parent / "prepared_dataset_detection" / "data.yaml").resolve()

def main():
    # pick a small model first (n = nano) for MPS
    model = YOLO("yolov12n.pt")
    print(f"Using data.yaml: {DATA_YAML}")

    results = model.train(
        data=str(DATA_YAML),          # absolute path to prepared_dataset/data.yaml
        epochs=150,                   # was 100; curves were still rising → give it more runway (early-stop still on)
        batch=1,                      # safe on MPS; try 4 if you ever hit OOM, or 16 if you have headroom
        imgsz=640,                    # try 960 later for small-object recall (reduce batch if needed)
        device="mps",                # use Apple Metal GPU; much faster than CPU
        amp=False,                    # key for MPS stability (avoids view/reshape runtime)
        half=False, 
        lr0=0.01,                    # conservative base LR for small dataset
        cos_lr=True,                  # smoother schedule; often helps convergence
        patience=50,                  # early stopping if val plateaus
        # Mild augs (uncomment to try after a stable baseline)
        # mosaic=0.5, mixup=0.05, copy_paste=0.1,
        project="runs",
        name="pothole_yolov12_mps",
        save=True
    )

    # optional: evaluate
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()