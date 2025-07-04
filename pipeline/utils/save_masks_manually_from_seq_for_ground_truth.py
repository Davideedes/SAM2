#!/usr/bin/env python3
# save_masks_interactively.py
"""
Interactively create (or skip) SAM2 point-masks for every image in a folder.

▶ Left-click  – positive point
▶ Right-click – negative point
▶ ENTER       – finish current image (skip if no points)
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Make sure Python can find the local 'sam2' package when the script is
# inside a sub-folder of your project.
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sam2.sam2_video_predictor import SAM2VideoPredictor  # noqa: E402

# ----------------------------------------------------------------------
# Helper for mask-preview (optional – remove if you don’t need previews)
# ----------------------------------------------------------------------
def _show_mask(mask, ax, color=np.array([1.0, 0.0, 0.0, 0.6])):
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def _show_points(points, labels, ax, size=375):
    if len(points) == 0:
        return
    points, labels = np.asarray(points), np.asarray(labels)
    pos = points[labels == 1]
    neg = points[labels == 0]
    if len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], c="lime", marker="*", s=size,
                   edgecolors="w", linewidths=1.25)
    if len(neg):
        ax.scatter(neg[:, 0], neg[:, 1], c="purple", marker="*", s=size,
                   edgecolors="w", linewidths=1.25)


# ----------------------------------------------------------------------
def main(img_dir: Path, mask_dir: Path, model_size: str = "tiny",
         recurse: bool = False):
    """Loop over all images in *img_dir*, collect clicks and save .npz."""
    img_dir = img_dir.expanduser().resolve()
    mask_dir = mask_dir.expanduser().resolve()
    mask_dir.mkdir(parents=True, exist_ok=True)

    # gather image paths
    exts = (".jpg", ".jpeg", ".png")
    if recurse:
        image_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
    else:
        image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    if not image_paths:
        print(f"No images found in {img_dir}")
        return

    # --------------------------------------------------------------
    # Resize helper – we use the smallest WxH across all frames
    # --------------------------------------------------------------
    sizes = [Image.open(p).size for p in image_paths]
    min_w = min(w for w, _ in sizes)
    min_h = min(h for _, h in sizes)
    target_size = (min_w, min_h)

    # --------------------------------------------------------------
    # Prepare SAM2 predictor once (dummy video with all frames)
    # --------------------------------------------------------------
    print("Loading SAM2 predictor …")
    predictor = SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")

    print("Generating temporary dummy video for SAM2 …")
    tmp = Path(tempfile.mkdtemp())
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB").resize(target_size, Image.LANCZOS)
        img.save(tmp / f"{idx:05d}.jpg")

    state = predictor.init_state(video_path=str(tmp))

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------
    for frame_idx, img_path in enumerate(image_paths):
        frame = np.array(Image.open(img_path).convert("RGB").resize(target_size,
                                                                    Image.LANCZOS))

        clicked_points, clicked_labels = [], []

        def _on_click(event):
            if event.inaxes is None:
                return
            if event.button not in (1, 3):       # 1 = LMB, 3 = RMB
                return
            clicked_points.append([int(event.xdata), int(event.ydata)])
            clicked_labels.append(1 if event.button == 1 else 0)
            # refresh preview
            ax.clear()
            ax.imshow(frame)
            if clicked_points:
                pts = np.asarray(clicked_points, np.float32)
                lbs = np.asarray(clicked_labels, np.int32)
                _, _, out_logits = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=pts,
                    labels=lbs,
                )
                _show_mask((out_logits[0] > 0).cpu().numpy(), ax)
                _show_points(pts, lbs, ax)
            ax.set_title(f"{img_path.name} – "
                         f"{len(clicked_points)} point(s)  (ENTER = next)")
            ax.axis("off")
            fig.canvas.draw_idle()

        def _on_key(event):
            if event.key == "enter":
                plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)
        ax.set_title(f"{img_path.name} – click to add/remove points "
                     "(ENTER = skip/save)")
        ax.axis("off")
        cid_click = fig.canvas.mpl_connect('button_press_event', _on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', _on_key)
        plt.show()

        # ----------------------------------------------------------
        # Save mask file only if we collected points
        # ----------------------------------------------------------
        if clicked_points:
            out_name = img_path.stem + ".npz"
            np.savez(mask_dir / out_name,
                     points=np.asarray(clicked_points, np.float32),
                     labels=np.asarray(clicked_labels, np.int32))
            print(f"✔ Saved mask: {out_name}")
        else:
            print("⤼ Skipped (no points)")

    # cleanup
    shutil.rmtree(tmp)
    print("\nAll done – temporary files removed.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively create SAM2 point-masks for every image in a folder.")
    parser.add_argument("img_dir", type=Path,
                        help="Folder that contains the input images.")
    parser.add_argument("--mask_dir", type=Path, default="training_pictures_masks",
                        help="Output folder for .npz files (default: ./training_pictures_masks).")
    parser.add_argument("--model_size", choices=["tiny", "small", "base_plus", "large"],
                        default="tiny", help="SAM2 weight variant (default: tiny).")


    args = parser.parse_args()
    main(args.img_dir, args.mask_dir, args.model_size)
