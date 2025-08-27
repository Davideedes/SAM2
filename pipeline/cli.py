from __future__ import annotations
import argparse
from .evaluate import run

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--model-size", choices=("tiny","small","base-plus","large","custom"), default="tiny")
    ap.add_argument("--seq-folder", required=True)
    ap.add_argument("--masks-folder")
    ap.add_argument("--cfg-path")   # nur bei --model-size custom
    ap.add_argument("--ckpt-path")  # nur bei --model-size custom

    # ------ NEU: IoU/GT-Optionen ------
    ap.add_argument("--gt-mask-dir", help="Ordner mit GT-Masken (.npz/.png/.jpg)")
    ap.add_argument("--gt-prefix", default="train_mask_", help="Prefix vor Bildstem in GT-Dateien")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU-Schwelle für lokales OK")
    ap.add_argument("--min-pixels", type=int, default=0, help="Vorhersagemasken <N Pixel werden ignoriert")

    args = ap.parse_args()

    run(
        n_train     = args.n_train,
        model_size  = args.model_size,
        seq_folder  = args.seq_folder,
        masks_folder= args.masks_folder,
        ckpt_path   = args.ckpt_path,
        cfg_path    = args.cfg_path,
        # NEU:
        gt_mask_dir = args.gt_mask_dir,
        gt_prefix   = args.gt_prefix,
        iou_thr     = args.iou_thr,
        min_pixels  = args.min_pixels,
    )

if __name__ == "__main__":
    main()
