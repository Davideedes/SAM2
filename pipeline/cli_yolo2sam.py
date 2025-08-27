# pipeline/cli_yolo2sam.py
import argparse
from .evaluate_yolo2sam import run_yolo2sam

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=("tiny","small","base-plus","large","custom"), default="tiny")
    ap.add_argument("--seq-folder", required=True, help="Ordner mit den Test-Bildern (JPG/PNG)")
    ap.add_argument("--labels-folder", required=True, help="Ordner mit YOLO .txt-Predictions (pro Bild)")
    ap.add_argument("--masks-folder", help="Zielordner für True-Positive-Masken (NPZ)")
    ap.add_argument("--cfg-path")   # nur bei --model-size custom
    ap.add_argument("--ckpt-path")  # nur bei --model-size custom
    ap.add_argument("--max-side", type=int, default=0, help="Longest side resize (0=off)")
    args = ap.parse_args()

    frames, segs, names = run_yolo2sam(
        model_size   = args.model_size,
        seq_folder   = args.seq_folder,
        labels_folder= args.labels_folder,
        masks_folder = args.masks_folder,
        ckpt_path    = args.ckpt_path,
        cfg_path     = args.cfg_path,
        max_side     = args.max_side,
    )
    # Optional: Visualisierung
    # from .browser import ImageBrowser
    # ImageBrowser(frames, segs, names)

if __name__ == "__main__":
    main()
