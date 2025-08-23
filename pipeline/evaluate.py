from __future__ import annotations
from pathlib import Path
import time, datetime, json, tempfile, os, shutil
import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor
from .config import load_ground_truth,load_input_sample_pictures, LOG_DIR, TRAIN_DIR, TRAIN_NPZ_MASK_DIR
from .vision import load_and_resize, save_mask_npz

GROUND_TRUTH = load_ground_truth()
TRAIN_SEQUENCE = load_input_sample_pictures()


def _new_run_dict(**kwargs) -> dict:
    """
    Erstellt ein neues Log-Dictionary zur Laufzeitüberwachung.

    Parameter
    ----------
    kwargs : dict
        Muss enthalten: model_size, n_train, seq_folder, masks_folder

    Returns
    -------
    dict : Leere Logstruktur mit Metadaten
    """
    return {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "model_size":  kwargs["model_size"],
        "n_train":     kwargs["n_train"],
        "seq_folder":  Path(kwargs["seq_folder"]).name,
        "masks_folder":Path(kwargs["masks_folder"]).name if kwargs["masks_folder"] else None,
        "expected_total": 0, "found_total": 0,
        "true_positives":0,"true_negatives":0,"false_negatives":0,"false_positives":0,
        "per_image": [], "runtime_seconds": 0.0,
    }

def run(
    n_train:int,
    model_size:str,
    seq_folder:str|Path,
    masks_folder:str|Path|None = None,
    ckpt_path:str|None = None,
    cfg_path:str|None  = None,
):
    """
Hauptlogik für Cross-Image-Transfer mit SAM2-Video-Predictor.

run(...)
-------
Parameters
----------
n_train : int
    Zahl der Trainingsframes (1–7).  
model_size : str
    "tiny" | "small" | "base" | "large" | "custom"  
seq_folder : str | Path
    Ordner mit Sequenzbildern.  
masks_folder : str | Path | None
    Zielordner für NPZ-Masken (optional).  
ckpt_path, cfg_path : str | None
    Lokaler Checkpoint & YAML (nur bei model_size=="custom").

Returns
-------
frames : list[np.ndarray]
segments : dict[int, dict[int, np.ndarray]]
names : list[str]
"""
    start = time.time()
    seq_folder  = Path(seq_folder)
    masks_folder = Path(masks_folder) if masks_folder else None
    if masks_folder:
       masks_folder.mkdir(parents=True, exist_ok=True)
    run_log     = _new_run_dict(n_train=n_train, model_size=model_size,
                                seq_folder=seq_folder, masks_folder=masks_folder)
    if n_train > len(TRAIN_SEQUENCE):
        raise ValueError(
            f"n_train ({n_train}) liegt über der definierten Sequenzlänge "
            f"({len(TRAIN_SEQUENCE)})"
        )
    train_names  = TRAIN_SEQUENCE[:n_train]
    train_paths  = [TRAIN_DIR / name for name in train_names]

    # Safety-Check: existieren alle Dateien?
    missing = [p for p in train_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Fehlende Trainingsbilder: {missing}")

    train_imgs  = [Image.open(p).convert("RGB") for p in train_paths]

    tgt_size    = min(i.width for i in train_imgs), min(i.height for i in train_imgs)
    train_np    = [np.array(i.resize(tgt_size, Image.LANCZOS)) for i in train_imgs]

    predictor = _make_predictor(model_size, ckpt_path, cfg_path)


    seq_paths   = sorted(p for p in seq_folder.iterdir()
                         if p.suffix.lower() in {".jpg",".jpeg",".png"})

    all_frames, all_segments = [], {}
    for idx, path in enumerate(seq_paths):
        frames = train_np + [load_and_resize(path, tgt_size)]
        with tempfile.TemporaryDirectory() as tmp:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(Path(tmp)/f"{i:05d}.jpg")
            state = predictor.init_state(video_path=tmp)
            _add_click_points(predictor, state, train_names)  

            segments = _propagate(predictor, state)
        all_frames.append(frames[-1])
        all_segments[len(all_frames)-1] = segments.get(len(frames)-1, {})

        _update_metrics(run_log, path.name, segments.get(len(frames)-1, {}), masks_folder)

    run_log["runtime_seconds"] = round(time.time()-start, 3)
    _write_log(run_log)
    return all_frames, all_segments, [p.name for p in seq_paths]

# --- Helfer ----------------------------------------------------
def _make_predictor(model_size:str, ckpt:str|None, cfg:str|None):
    """
    Erstellt den passenden SAM2VideoPredictor (pretrained oder lokal).

    Returns
    -------
    SAM2VideoPredictor
    """
    if model_size != "custom":
        return SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")
    if not (ckpt and cfg):
        raise ValueError("Für model_size=custom müssen --cfg-path und --ckpt-path gesetzt sein")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
    return predictor


def _add_click_points(predictor, state, train_names:list[str]):
    """
    Fügt dem Predictor die Maskenpunkte der Trainingsbilder hinzu.
    """
    for i, name in enumerate(train_names):

        data = np.load(TRAIN_NPZ_MASK_DIR/f"train_mask_{os.path.splitext(name)[0]}.npz")
        if data["points"].size:
            predictor.add_new_points_or_box(state, i, 1, data["points"], data["labels"])


def _propagate(predictor, state):
    """
    Führt die Maskenpropagation über alle Frames durch.

    Returns
    -------
    dict[int, dict[int, np.ndarray]]
        segment_id → Maske für jedes Frame
    """
    out = {}
    for f_idx, ids, logits in predictor.propagate_in_video(state):
        out[f_idx] = {oid:(logits[i]>0).cpu().numpy() for i,oid in enumerate(ids)}
    return out
def _update_metrics(log:dict, img_name:str, segs:dict, masks_folder:Path|None):
    """
    Aktualisiert die Auswertungsmetriken und speichert ggf. die Maske.
    """
    stem = Path(img_name).stem
    exp  = GROUND_TRUTH.get(stem, False)
    found = bool(1 in segs and segs[1].sum() > 0)

    if exp and found and masks_folder:
        save_mask_npz(segs[1], masks_folder/f"{Path(img_name).stem}.npz")

    log["per_image"].append({"img_name":img_name,"expected":exp,"found":found,
                             "type":("tp" if exp and found else
                                     "tn" if not exp and not found else
                                     "fn" if exp and not found else "fp")})
    if exp:   log["expected_total"] += 1
    if found: log["found_total"]    += 1
    if   exp and found:      log["true_positives"]  += 1
    elif exp and not found:  log["false_negatives"] += 1
    elif not exp and found:  log["false_positives"] += 1
    else:                    log["true_negatives"]  += 1

def _write_log(log:dict):
    """
    Speichert das Log als JSON im Log-Verzeichnis.
    """
    ts = log["timestamp"].replace(":","-").replace("T","_")
    name = f"Model{log['model_size']}_nTrain{log['n_train']}_{ts}.json"
    with open(LOG_DIR/name,"w",encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print("💾 Log gespeichert:", LOG_DIR/name)
