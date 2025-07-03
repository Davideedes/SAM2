import argparse
from .evaluate import run
from .browser import ImageBrowser

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--model-size", choices=("tiny","base","large"), default="tiny")
    ap.add_argument("--seq-folder", required=True)
    ap.add_argument("--masks-folder")        # optional
    args = ap.parse_args()

    frames, segs, names = run(args.n_train, args.model_size,
                              args.seq_folder, args.masks_folder)
    #ImageBrowser(frames, segs, names)

if __name__ == "__main__":
    main()
