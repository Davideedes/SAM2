import argparse
from .evaluate_yolo2sam import run_yolo2sam

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=("tiny","small","base-plus","large","custom"), default="tiny")
    ap.add_argument("--seq-folder", required=True)
    ap.add_argument("--labels-folder", required=True)
    ap.add_argument("--masks-folder")
    ap.add_argument("--cfg-path")
    ap.add_argument("--ckpt-path")
    ap.add_argument("--max-side", type=int, default=0)
    # Neu:
    ap.add_argument("--gt-mask-dir")
    ap.add_argument("--gt-prefix", default="train_mask_")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--min-pixels", type=int, default=0)
    args = ap.parse_args()

    run_yolo2sam(
        model_size    = args.model_size,
        seq_folder    = args.seq_folder,
        labels_folder = args.labels_folder,
        masks_folder  = args.masks_folder,
        ckpt_path     = args.ckpt_path,
        cfg_path      = args.cfg_path,
        max_side      = args.max_side,
        # Neu:
        gt_mask_dir   = args.gt_mask_dir,
        gt_prefix     = args.gt_prefix,
        iou_thr       = args.iou_thr,
        min_pixels    = args.min_pixels,
    )

if __name__ == "__main__":
    main()
