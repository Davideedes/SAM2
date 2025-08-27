from __future__ import annotations
from pathlib import Path
import time, datetime, json, tempfile, os
import numpy as np
import torch
from PIL import Image

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from .config import (
    load_ground_truth,
    load_input_sample_pictures,
    LOG_DIR,
    TRAIN_DIR,
    TRAIN_NPZ_MASK_DIR,
)
from .vision import load_and_resize, save_mask_npz

GROUND_TRUTH  = load_ground_truth()
TRAIN_SEQUENCE= load_input_sample_pictures()

# ---------- IoU-Helpers ----------
def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred{pred.shape} vs gt{gt.shape}")
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def _load_gt_mask_any(gt_dir: Path, stem: str, prefix: str = "train_mask_") -> np.ndarray | None:
    """
    Lädt GT-Maske (binär) aus gt_dir. Unterstützt:
      - <stem>.npz / .png / .jpg
      - <prefix><stem>.npz / .png / .jpg (z.B. 'train_mask_12345.npz')
    NPZ: bevorzugt key 'mask', sonst erstes Array.
    """
    candidates = [stem, f"{prefix}{stem}"]

    # NPZ
    for base in candidates:
        npz = gt_dir / f"{base}.npz"
        if npz.exists():
            d = np.load(npz)
            if "mask" in d:
                arr = d["mask"]
            elif "arr_0" in d:
                arr = d["arr_0"]
            else:
                keys = list(d.keys())
                arr = d[keys[0]] if keys else None
            if arr is None:
                return None
            return (arr > 0).astype(bool)

    # PNG/JPG
    for base in candidates:
        for ext in (".png", ".jpg", ".jpeg"):
            p = gt_dir / f"{base}{ext}"
            if p.exists():
                im = Image.open(p).convert("L")
                return (np.array(im) > 0)

    return None

# ---------- Log-Struktur ----------
def _new_run_dict(**kwargs) -> dict:
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_size": kwargs["model_size"],
        "n_train": kwargs["n_train"],
        "seq_folder": Path(kwargs["seq_folder"]).name,
        "masks_folder": Path(kwargs["masks_folder"]).name if kwargs["masks_folder"] else None,

        # bestehende binäre Bild-Eval
        "expected_total": 0, "found_total": 0,
        "true_positives": 0, "true_negatives": 0,
        "false_negatives": 0,"false_positives": 0,

        # NEU: IoU/Lokalisierung
        "gt_mask_dir": str(kwargs.get("gt_mask_dir") or ""),
        "gt_prefix": kwargs.get("gt_prefix") or "train_mask_",
        "iou_threshold": float(kwargs.get("iou_thr") or 0.5),
        "min_pixels": int(kwargs.get("min_pixels") or 0),

        "loc_true_positives": 0,
        "loc_false_negatives": 0,
        "pred_with_mask": 0,
        "iou_sum": 0.0,
        "iou_count": 0,

        "per_image": [],
        "runtime_seconds": 0.0,
    }

# ---------- Hauptlauf ----------
def run(
    n_train: int,
    model_size: str,
    seq_folder: str | Path,
    masks_folder: str | Path | None = None,
    ckpt_path: str | None = None,
    cfg_path: str | None = None,
    # NEU: IoU-Optionen
    gt_mask_dir: str | Path | None = None,
    gt_prefix: str = "train_mask_",
    iou_thr: float = 0.5,
    min_pixels: int = 0,
):
    start = time.time()

    seq_folder  = Path(seq_folder)
    masks_folder= Path(masks_folder) if masks_folder else None
    if masks_folder:
        masks_folder.mkdir(parents=True, exist_ok=True)
    gt_mask_dir = Path(gt_mask_dir) if gt_mask_dir else None

    run_log = _new_run_dict(
        n_train=n_train, model_size=model_size,
        seq_folder=seq_folder, masks_folder=masks_folder,
        gt_mask_dir=gt_mask_dir, gt_prefix=gt_prefix,
        iou_thr=iou_thr, min_pixels=min_pixels,
    )

    if n_train > len(TRAIN_SEQUENCE):
        raise ValueError(f"n_train ({n_train}) > Sequenzlänge ({len(TRAIN_SEQUENCE)})")

    # Trainingsframes laden (für SAM2-VideoPrompting)
    train_names = TRAIN_SEQUENCE[:n_train]
    train_paths = [TRAIN_DIR / name for name in train_names]
    missing = [p for p in train_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Fehlende Trainingsbilder: {missing}")

    train_imgs = [Image.open(p).convert("RGB") for p in train_paths]
    tgt_size = (min(i.width for i in train_imgs), min(i.height for i in train_imgs))
    train_np = [np.array(i.resize(tgt_size, Image.LANCZOS)) for i in train_imgs]

    predictor = _make_predictor(model_size, ckpt_path, cfg_path)

    seq_paths = sorted(p for p in seq_folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    all_frames, all_segments = [], {}

    for idx, path in enumerate(seq_paths):
        # Video = n_train Frames + 1 Testframe (auf tgt_size)
        frames = train_np + [load_and_resize(path, tgt_size)]
        with tempfile.TemporaryDirectory() as tmp:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(Path(tmp)/f"{i:05d}.jpg")
            state = predictor.init_state(video_path=tmp)

        _add_click_points(predictor, state, train_names)
        segments = _propagate(predictor, state)

        # Testframe = letztes Frame
        test_fidx = len(frames) - 1
        segs_last = segments.get(test_fidx, {})

        # Binäre Bild-Eval + optional Speichern
        all_frames.append(frames[-1])
        all_segments[len(all_frames)-1] = segs_last
        _update_metrics(run_log, path.name, segs_last, masks_folder)

        # -------- NEU: IoU/Lokalisierung gegen GT --------
        iou_val = None
        loc_ok  = False

        if gt_mask_dir is not None:
            gt = _load_gt_mask_any(gt_mask_dir, path.stem, prefix=gt_prefix)
            if gt is not None:
                # GT auf tgt_size bringen (Nearest → binär bleibt erhalten)
                if gt.shape != tgt_size[::-1]:
                    gt_img = Image.fromarray(gt.astype(np.uint8)*255)
                    gt_img = gt_img.resize(tgt_size, Image.NEAREST)
                    gt = (np.array(gt_img) > 0)

                # Vorhersage (Only-SAM: Objekt-ID 1)
                pred = segs_last.get(1, None)
                if pred is not None and pred.ndim == 3:
                    pred = np.squeeze(pred)
                # min_pixels-Filter
                if pred is None or (min_pixels > 0 and pred.sum() < min_pixels):
                    iou_val = 0.0
                else:
                    iou_val = _mask_iou(pred.astype(bool), gt.astype(bool))
                    run_log["pred_with_mask"] += 1

                # Lokales OK nur für GT-positive Bilder
                if GROUND_TRUTH.get(path.stem, False):
                    if iou_val >= iou_thr:
                        run_log["loc_true_positives"] += 1
                    else:
                        run_log["loc_false_negatives"] += 1
                    run_log["iou_sum"] += float(iou_val)
                    run_log["iou_count"] += 1
                    loc_ok = (iou_val >= iou_thr)

        # per_image um IoU-Felder ergänzen
        run_log["per_image"][-1].update({
            "iou_max": iou_val,
            "loc_ok": bool(loc_ok),
        })

    # Summaries
    run_log["runtime_seconds"] = round(time.time()-start, 3)
    run_log["mean_iou_gt_pos"] = (run_log["iou_sum"]/run_log["iou_count"]) if run_log["iou_count"]>0 else None
    gt_pos = sum(1 for r in run_log["per_image"] if r["expected"])
    run_log["loc_recall"] = (run_log["loc_true_positives"]/gt_pos) if gt_pos>0 and run_log["gt_mask_dir"] else None
    if run_log["pred_with_mask"]>0 and run_log["gt_mask_dir"]:
        loc_ok_count = sum(1 for r in run_log["per_image"] if r.get("loc_ok"))
        run_log["loc_precision"] = loc_ok_count / run_log["pred_with_mask"]
    else:
        run_log["loc_precision"] = None

    _write_log(run_log)
    return all_frames, all_segments, [p.name for p in seq_paths]

# --- Helfer ----------------------------------------------------
def _make_predictor(model_size:str, ckpt:str|None, cfg:str|None):
    if model_size != "custom":
        return SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")
    if not (ckpt and cfg):
        raise ValueError("Für model_size=custom müssen --cfg-path und --ckpt-path gesetzt sein")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
    return predictor

def _add_click_points(predictor, state, train_names:list[str]):
    # lädt Trainings-Klicks (points/labels) und fügt sie als Prompts hinzu
    for i, name in enumerate(train_names):
        stem = os.path.splitext(name)[0]
        data = np.load(TRAIN_NPZ_MASK_DIR / f"train_mask_{stem}.npz")
        pts = data["points"] if "points" in data else np.zeros((0,2),dtype=np.float32)
        lbl = data["labels"] if "labels" in data else np.zeros((0,),dtype=np.int32)
        if pts.size:
            predictor.add_new_points_or_box(state, i, 1, pts, lbl)

def _propagate(predictor, state):
    out = {}
    for f_idx, ids, logits in predictor.propagate_in_video(state):
        out[f_idx] = {int(oid):(logits[i] > 0).cpu().numpy() for i,oid in enumerate(ids)}
    return out

def _update_metrics(log:dict, img_name:str, segs:dict, masks_folder:Path|None):
    stem = Path(img_name).stem
    exp  = GROUND_TRUTH.get(stem, False)
    pred = segs.get(1, None)
    found = bool(pred is not None and pred.sum() > 0)

    if exp and found and masks_folder is not None:
        save_mask_npz(pred, masks_folder/f"{stem}.npz")

    log["per_image"].append({
        "img_name": img_name,
        "expected": exp,
        "found": found,
        "n_objects": len(segs),
    })
    if exp:   log["expected_total"] += 1
    if found: log["found_total"]    += 1
    if   exp and found:      log["true_positives"]  += 1
    elif exp and not found:  log["false_negatives"] += 1
    elif not exp and found:  log["false_positives"] += 1
    else:                    log["true_negatives"]  += 1

def _write_log(log:dict):
    ts = log["timestamp"].replace(":","-").replace("T","_")
    name = f"Model{log['model_size']}_nTrain{log['n_train']}_{ts}.json"
    with open(LOG_DIR/name,"w",encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print("💾 Log gespeichert:", LOG_DIR/name)
