from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Iterable

from .yolo_seg_infer import run_yolo_seg_folder

def parse_classes(s: Optional[str]) -> Optional[Iterable[int]]:
    if not s: return None
    return [int(x) for x in s.replace(",", " ").split() if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="z.B. yolov8n-seg.pt oder eigener .pt Pfad")
    ap.add_argument("--seq-folder", required=True)
    ap.add_argument("--masks-folder", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.60, help="NMS IoU")
    ap.add_argument("--device", default=0, help="GPU-Index oder 'cpu'")
    ap.add_argument("--classes", default=None, help="z.B. '0,2,5'")
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--save-png-instances", action="store_true")
    ap.add_argument("--merge", choices=("largest","sum","none"), default="largest")

    # Eval / IoU
    ap.add_argument("--gt-mask-dir")
    ap.add_argument("--gt-prefix", default="train_mask_")
    ap.add_argument("--iou-thr", type=float, default=0.5)

    args = ap.parse_args()

    run_yolo_seg_folder(
        weights=args.weights,
        source_dir=args.seq_folder,
        masks_out=args.masks_folder,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=parse_classes(args.classes),
        min_pixels=args.min_pixels,
        save_png_instances=args.save_png_instances,
        merge_strategy=args.merge,
        gt_mask_dir=args.gt_mask_dir,
        gt_prefix=args.gt_prefix,
        iou_thr=args.iou_thr,
    )

if __name__ == "__main__":
    main()
