"""
Command-line interface zum Ausführen der SAM2-Video-Evaluation.

Parameter
---------
--model-size {tiny,small,base,large,custom}
    *tiny/small/base/large*  →  lädt vortrainiertes Modell von Hugging-Face  
    *custom*                 →  lädt lokales Modell (--cfg-path + --ckpt-path nötig)

--n-train INT
    Anzahl Trainingsbilder (1 … len(input_pictures_sequence.json))

--seq-folder PATH
    Ordner mit den Test-Bildern (JPG/PNG).  
    Relativ zum Projektroot oder absolut.

--masks-folder PATH (optional)
    Zielordner für True-Positive-Masken im NPZ-Format.  
    Wird nur angelegt/beschrieben, wenn angegeben.

--cfg-path PATH (nur bei --model-size custom)
    YAML-Config des lokalen Checkpoints.

--ckpt-path PATH (nur bei --model-size custom)
    Gewichts-Datei (*.pt) des lokalen Checkpoints.


Beispiele
---------
▶  Pretrained-Tiny, 3 Trainingsbilder, Masken speichern  
    python3 -m pipeline.cli \
        --model-size tiny \
        --n-train 3 \
        --seq-folder pipeline/resources/sequence_to_test_1 \
        --masks-folder results/tiny_n3

▶  Custom-Checkpoint  
    python3 -m pipeline.cli \
        --model-size custom \
        --n-train 5 \
        --seq-folder pipeline/resources/sequence_to_test_1 \
        --cfg-path configs/sam2.1/sam2.1_hiera_t.yaml \
        --ckpt-path checkpoints/sam2.1_hiera_tiny.pt
"""

import argparse
from .evaluate import run
from .browser import ImageBrowser

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--model-size", choices=("tiny","small", "base","large", "custom"), default="tiny")
    ap.add_argument("--seq-folder", required=True)
    ap.add_argument("--masks-folder")  
    ap.add_argument("--cfg-path")    # nur nötig bei --model-size custom
    ap.add_argument("--ckpt-path")   # "      # optional
    args = ap.parse_args()

    frames, segs, names = run(
        n_train      = args.n_train,
        model_size   = args.model_size,
        seq_folder   = args.seq_folder,
        masks_folder = args.masks_folder,
        ckpt_path    = args.ckpt_path,
        cfg_path     = args.cfg_path,
    )
    #ImageBrowser(frames, segs, names)

if __name__ == "__main__":
    main()

# python3 -m pipeline.cli \
#     --model-size custom \
#     --n-train 5 \
#     --seq-folder pipeline/resources/sequence_to_test_1 \
#     --cfg-path configs/sam2.1/sam2.1_hiera_t.yaml \
#     --ckpt-path checkpoints/sam2.1_hiera_tiny.pt