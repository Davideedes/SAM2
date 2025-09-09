from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, Literal, Dict, Any
import time, datetime, json
import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO

from .config import load_ground_truth, LOG_DIR
from .vision import save_mask_npz

GROUND_TRUTH = load_ground_truth()


def _new_run_dict(**kwargs) -> dict:
    return {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "mode":        "yolo-only",
        "weights":     kwargs["weights"],
        "seq_folder":  Path(kwargs["seq_folder"]).name,
        "masks_folder": Path(kwargs["masks_folder"]).name if kwargs["masks_folder"] else None,
        "imgsz": kwargs.get("imgsz"),
        "conf":  kwargs.get("conf"),
        "iou_nms": kwargs.get("iou"),
        "device": kwargs.get("device"),
        "classes": list(kwargs.get("classes") or []),

        "expected_total": 0, "found_total": 0,
        "true_positives":0,"true_negatives":0,"false_negatives":0,"false_positives":0,

        # IoU / Lokalisation
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


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred{pred.shape} vs gt{gt.shape}")
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _load_gt_mask_any(gt_dir: Path, stem: str, prefix: str = "train_mask_") -> np.ndarray | None:
    # identisch zur Logik in deiner SAM-Pipeline
    candidates = [stem, f"{prefix}{stem}"]
    for base in candidates:
        npz = gt_dir / f"{base}.npz"
        if npz.exists():
            d = np.load(npz)
            arr = d["mask"] if "mask" in d else d.get("arr_0")
            if arr is None:
                keys = list(d.keys())
                arr = d[keys[0]] if keys else None
            if arr is None:
                return None
            return (arr > 0).astype(bool)
    for base in candidates:
        for ext in (".png", ".jpg", ".jpeg"):
            p = gt_dir / f"{base}{ext}"
            if p.exists():
                im = Image.open(p).convert("L")
                return (np.array(im) > 0)
    return None


def _write_log(log:dict, weights:str):
    ts = log["timestamp"].replace(":","-").replace("T","_")
    name = f"YOLOONLY_{Path(weights).stem}_{ts}.json"

    log_dir = LOG_DIR / "only_yolo" 
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / name, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print("💾 Log gespeichert:", LOG_DIR / name)


def _result_masks_to_numpy(result) -> np.ndarray | None:
    """
    Gibt (N,H,W) float/uint8 zurück, *möglichst* bereits in Original-Bildgröße.
    Ultralytics >=8 liefert result.masks.data (N,h,w) float in [0,1] + upsampling intern.
    Fallback: wir resizen selbst auf result.orig_shape (H,W).
    """
    mobj = getattr(result, "masks", None)
    if mobj is None:
        return None
    data = getattr(mobj, "data", None)
    if data is None:
        return None
    arr = data.detach().cpu().numpy()  # (N,h,w), float32 in [0,1]
    H, W = result.orig_shape  # (H,W)
    if arr.shape[-2:] != (H, W):
        # auf Originalgröße bringen
        out = []
        for a in arr:
            im = Image.fromarray((a * 255).astype(np.uint8))
            im = im.resize((W, H), Image.NEAREST)
            out.append(np.array(im).astype(np.float32) / 255.0)
        arr = np.stack(out, axis=0)
    return arr  # (N,H,W) float in [0,1]


def run_yolo_seg_folder(
    weights: str,
    source_dir: str | Path,
    masks_out: str | Path | None = None,
    imgsz: int = 1024,
    conf: float = 0.20,
    iou: float = 0.60,
    device: int | str = 0,
    classes: Optional[Iterable[int]] = None,
    min_pixels: int = 0,
    save_png_instances: bool = False,
    merge_strategy: Literal["largest", "sum", "none"] = "largest",
    # Eval/Log
    gt_mask_dir: str | Path | None = None,
    gt_prefix: str = "train_mask_",
    iou_thr: float = 0.5,
) -> Dict[str, Any]:
    """
    Führt YOLO-Seg Inferenz über alle Bilder in source_dir aus.
    - Speichert je Bild eine *binäre* Maske als NPZ (standard: größte Instanz) unter masks_out/<stem>.npz.
    - Optional: alle Instanzmasken zusätzlich als PNG (Debug).
    - Optional: Vergleich gg. GT-Masken (IoU, Lokalisierung).
    """
    t0 = time.time()
    source_dir = Path(source_dir)
    if masks_out:
        masks_out = Path(masks_out); masks_out.mkdir(parents=True, exist_ok=True)
    gt_mask_dir = Path(gt_mask_dir) if gt_mask_dir else None

    log = _new_run_dict(
        weights=weights, seq_folder=source_dir,
        masks_folder=masks_out, imgsz=imgsz, conf=conf, iou=iou, device=device,
        classes=classes, gt_mask_dir=gt_mask_dir, gt_prefix=gt_prefix,
        iou_thr=iou_thr, min_pixels=min_pixels
    )

    model = YOLO(weights)
    results = model.predict(
        source=str(source_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        stream=True,
        verbose=False,
        save=False, save_txt=False
    )

    for r in results:
        img_name = Path(r.path).name
        stem = Path(r.path).stem
        H, W = r.orig_shape
        exp = GROUND_TRUTH.get(stem, False)

        masks_arr = _result_masks_to_numpy(r)  # (N,H,W) float or None
        found = False
        saved_mask = None

        if masks_arr is not None and masks_arr.shape[0] > 0:
            # Threshold auf binär, min_pixels filter
            bin_masks = (masks_arr > 0.5)
            if min_pixels > 0:
                keep = [i for i, m in enumerate(bin_masks) if int(m.sum()) >= min_pixels]
                bin_masks = bin_masks[keep] if keep else np.zeros((0, H, W), dtype=bool)

            if bin_masks.shape[0] > 0:
                found = True

                if save_png_instances:
                    inst_dir = (masks_out / "instances_png") if masks_out else None
                    if inst_dir:
                        inst_dir.mkdir(parents=True, exist_ok=True)
                        for k, m in enumerate(bin_masks, start=1):
                            Image.fromarray((m.astype(np.uint8) * 255)).save(inst_dir / f"{stem}_obj{k:02d}.png")

                if merge_strategy == "largest":
                    areas = [int(m.sum()) for m in bin_masks]
                    best = int(np.argmax(areas))
                    saved_mask = bin_masks[best]
                elif merge_strategy == "sum":
                    saved_mask = np.any(bin_masks, axis=0)
                else:  # "none" → wenn mehrere, nimm largest für das Standard-<stem>.npz, Rest optional separat
                    areas = [int(m.sum()) for m in bin_masks]
                    best = int(np.argmax(areas))
                    saved_mask = bin_masks[best]
                    if masks_out:
                        # Zusätzlich alle Instanzen als eigene npz speichern
                        inst_npz_dir = masks_out / "instances_npz"; inst_npz_dir.mkdir(parents=True, exist_ok=True)
                        for k, m in enumerate(bin_masks, start=1):
                            np.savez_compressed(inst_npz_dir / f"{stem}_obj{k:02d}.npz", mask=m.astype(np.uint8))

        # speichern (eine Maske pro Bild)
        if masks_out and saved_mask is not None:
            save_mask_npz(saved_mask, masks_out / f"{stem}.npz")

        # Bildklassifikation + Zähler
        log["per_image"].append({
            "img_name": img_name,
            "expected": bool(exp),
            "found": bool(found),
            "n_instances": int(masks_arr.shape[0]) if masks_arr is not None else 0,
            "iou_max": None,
            "loc_ok": False,
        })
        if exp:   log["expected_total"] += 1
        if found: log["found_total"]    += 1
        if   exp and found:      log["true_positives"]  += 1
        elif exp and not found:  log["false_negatives"] += 1
        elif not exp and found:  log["false_positives"] += 1
        else:                    log["true_negatives"]  += 1

        # IoU / Lokalisation gg. GT (falls vorhanden)
        if gt_mask_dir is not None:
            gt = _load_gt_mask_any(gt_mask_dir, stem, prefix=gt_prefix)
            if gt is not None:
                # Sicherheit: Größe matchen
                if gt.shape != (H, W):
                    gt_img = Image.fromarray(gt.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
                    gt = (np.array(gt_img) > 0)

                if masks_arr is not None and masks_arr.shape[0] > 0:
                    bin_masks = (masks_arr > 0.5)
                    if min_pixels > 0:
                        bin_masks = np.stack([m for m in bin_masks if int(m.sum()) >= min_pixels], axis=0) \
                                    if any(int(m.sum()) >= min_pixels for m in bin_masks) else np.zeros((0, H, W), bool)
                    iou_max = max((_mask_iou(m, gt) for m in bin_masks), default=0.0) if bin_masks.size else 0.0
                else:
                    iou_max = 0.0

                log["per_image"][-1]["iou_max"] = float(iou_max)
                if exp:
                    if iou_max >= iou_thr:
                        log["loc_true_positives"] += 1
                    else:
                        log["loc_false_negatives"] += 1
                    log["iou_sum"] += float(iou_max)
                    log["iou_count"] += 1
                log["per_image"][-1]["loc_ok"] = bool(iou_max >= iou_thr)

        if found:
            log["pred_with_mask"] += 1

    # Summaries
    log["runtime_seconds"] = round(time.time() - t0, 3)
    log["mean_iou_gt_pos"] = (log["iou_sum"] / log["iou_count"]) if log["iou_count"] > 0 else None
    gt_pos = sum(1 for r in log["per_image"] if r["expected"])
    if gt_pos > 0 and log["gt_mask_dir"]:
        log["loc_recall"] = log["loc_true_positives"] / gt_pos
    else:
        log["loc_recall"] = None
    if log["pred_with_mask"] > 0 and log["gt_mask_dir"]:
        loc_ok_count = sum(1 for r in log["per_image"] if r.get("loc_ok"))
        log["loc_precision"] = loc_ok_count / log["pred_with_mask"]
    else:
        log["loc_precision"] = None

    _write_log(log, weights)
    return log
