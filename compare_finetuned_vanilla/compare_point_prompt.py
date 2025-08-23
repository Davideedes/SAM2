#!/usr/bin/env python3
"""
compare_point_prompt.py
-----------------------

Compare vanilla vs fine-tuned SAM-2 tiny using a single point-click
(with Matplotlib ginput, avoids Qt/XCB issues).

Usage:
  python3 compare_finetuned_vanilla/compare_point_prompt.py     --img compare_finetuned_vanilla/3360951680713425.jpg     --cfg sam2_hiera_t               --vanilla-ckpt sam2.1_hiera_tiny.pt      --finetuned-ckpt checkpoints/fine_tuned_sam2_6500.pt     --device cuda

Make sure you have python3-tk installed (TkAgg backend).
"""

import argparse
import os
import numpy as np
import torch

# Force TkAgg backend (requires python3-tk)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2


def masks_to_segmap(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Greedy merge of SAM masks → seg map with label ids 1..K."""
    masks = np.asarray(masks)
    scores = np.asarray(scores)
    # Ensure mask batch dimension
    if masks.ndim == 2:
        masks = masks[None, ...]
        scores = np.atleast_1d(scores)

    ordered = masks[scores.argsort()[::-1]]
    seg = np.zeros_like(ordered[0], np.uint8)
    occupied = np.zeros_like(seg, bool)

    for i, m in enumerate(ordered):
        m_bin = m > 0
        total = m_bin.sum()
        # Skip empty masks or heavy overlap
        if total == 0:
            continue
        overlap = (m_bin & occupied).sum() / float(total)
        if overlap > 0.15:
            continue
        keep = m_bin & ~occupied
        seg[keep] = i + 1
        occupied |= keep
    # Guarantee 2D output
    return seg.reshape(seg.shape[-2], seg.shape[-1])


def get_click_matplotlib(img_rgb: np.ndarray):
    """
    Display `img_rgb` in a Matplotlib window and capture one click.
    Returns (x, y) image coordinates.
    """
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Click once to segment")
    ax.axis('off')
    coords = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not coords:
        raise RuntimeError("No click detected—please click once on the image.")
    x, y = coords[0]
    return int(x), int(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",           required=True, help="Path to RGB image")
    parser.add_argument("--cfg",           required=True, help="Config name (e.g. sam2_hiera_t)")
    parser.add_argument("--vanilla-ckpt",  required=True, help="Path to vanilla tiny .pt checkpoint")
    parser.add_argument("--finetuned-ckpt",required=True, help="Path to fine-tuned .pt checkpoint")
    parser.add_argument("--device",        default="cuda", choices=["cuda","cpu"])
    args = parser.parse_args()

    # Validate file paths
    for path in (args.img, args.vanilla_ckpt, args.finetuned_ckpt):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    device = torch.device(args.device)

    # Load image via OpenCV and convert to RGB contiguous
    bgr = cv2.imread(args.img)
    if bgr is None:
        raise RuntimeError(f"Failed to read image {args.img}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)

    # Build & load both SAM models
    sam_v = build_sam2(args.cfg, ckpt_path=args.vanilla_ckpt, device=device)
    sam_f = build_sam2(args.cfg, ckpt_path=args.finetuned_ckpt, device=device)
    pred_v = SAM2ImagePredictor(sam_v)
    pred_f = SAM2ImagePredictor(sam_f)

    # Capture user click
    x, y = get_click_matplotlib(rgb)
    pts = np.array([[[x, y]]], dtype=np.float32)
    lbl = np.ones((1,1), dtype=np.int64)

    # Run predictions
    with torch.no_grad():
        pred_v.set_image(rgb)
        m_v, s_v, _ = pred_v.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
        pred_f.set_image(rgb)
        m_f, s_f, _ = pred_f.predict(point_coords=pts, point_labels=lbl, multimask_output=True)

    # Merge masks to segmentation maps
    seg_v = masks_to_segmap(m_v[0], s_v[0])
    seg_f = masks_to_segmap(m_f[0], s_f[0])

    # Display comparison figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(rgb)
    axs[0].scatter([x], [y], c='r', s=40)
    axs[0].set_title("Image & click")
    axs[0].axis('off')

    axs[1].imshow(seg_v, cmap='jet')
    axs[1].set_title("Vanilla SAM-2")
    axs[1].axis('off')

    axs[2].imshow(seg_f, cmap='jet')
    axs[2].set_title("Fine-tuned SAM-2")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
