from __future__ import annotations
from pathlib import Path
import time, datetime, json, tempfile, os
import numpy as np
import torch
from PIL import Image

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from .config import load_ground_truth, LOG_DIR
from .vision import save_mask_npz
from .boxes_io import read_yolo_txt, yolo_xywhn_to_xyxy_pixels

GROUND_TRUTH = load_ground_truth()

# ---------- IoU-Helfer ----------
# ---------- IoU-Helfer ----------

def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred{pred.shape} vs gt{gt.shape}")
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def _load_gt_mask_any(gt_dir: Path, stem: str, prefix: str = "train_mask_") -> np.ndarray | None:
    """
    Lädt GT-Maske (binär) aus gt_dir.
    Unterstützt:
      - <stem>.npz / .png / .jpg
      - <prefix><stem>.npz / .png / .jpg   (z.B. 'train_mask_12345.npz')
    """
    candidates = [stem, f"{prefix}{stem}"]

    # 1) NPZ
    for base in candidates:
        npz = gt_dir / f"{base}.npz"
        if npz.exists():
            d = np.load(npz)
            # Bevorzugt 'mask', sonst erstes Array (arr_0), ansonsten None
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

    # 2) Bild
    for base in candidates:
        for ext in (".png", ".jpg", ".jpeg"):
            p = gt_dir / f"{base}{ext}"
            if p.exists():
                im = Image.open(p).convert("L")
                return (np.array(im) > 0)

    return None


# ---------- Logging-Struktur ----------
def _new_run_dict(**kwargs) -> dict:
    return {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "mode":        "yolo2sam",
        "model_size":  kwargs["model_size"],
        "seq_folder":  Path(kwargs["seq_folder"]).name,
        "labels_folder": Path(kwargs["labels_folder"]).name,
        "masks_folder": Path(kwargs["masks_folder"]).name if kwargs["masks_folder"] else None,
        "expected_total": 0, "found_total": 0,
        "true_positives":0,"true_negatives":0,"false_negatives":0,"false_positives":0,
        # IoU/Lokalisierung
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

def _make_predictor(model_size:str, ckpt:str|None, cfg:str|None):
    if model_size != "custom":
        return SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")
    if not (ckpt and cfg):
        raise ValueError("Für model_size=custom müssen --cfg-path und --ckpt-path gesetzt sein")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
    return predictor

def _add_yolo_boxes(predictor, state, frame_index:int, boxes_xyxy:list[tuple[int,int,int,int]]):
    for j, (x1,y1,x2,y2) in enumerate(boxes_xyxy, start=1):
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_index,
            obj_id=j,
            points=np.empty((0, 2), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            clear_old_points=True,
            normalize_coords=True,
            box=box
        )

def _propagate_single_frame(predictor, frame_np: np.ndarray):
    with tempfile.TemporaryDirectory() as tmp:
        Image.fromarray(frame_np).save(Path(tmp)/"00000.jpg")
        state = predictor.init_state(video_path=tmp)
        return state

# ---------- bestehende Bild-Eval (binär) ----------

def _update_metrics_multiids(log:dict, img_name:str, segs:dict, masks_folder:Path|None):
    stem = Path(img_name).stem
    exp  = GROUND_TRUTH.get(stem, False)
    areas = {oid:int(m.sum()) for oid,m in segs.items()} if segs else {}
    found = any(a > 0 for a in areas.values())

    if exp and found and masks_folder and areas:
        best_id = max(areas, key=areas.get)
        save_mask_npz(segs[best_id], masks_folder/f"{stem}.npz")

    log["per_image"].append({
        "img_name": img_name,
        "expected": exp,
        "found": found,
        "n_objects": len(segs),
        # IoU-Felder werden später in-place ergänzt, falls GT-Maske vorhanden
    })
    if exp:   log["expected_total"] += 1
    if found: log["found_total"]    += 1
    if   exp and found:      log["true_positives"]  += 1
    elif exp and not found:  log["false_negatives"] += 1
    elif not exp and found:  log["false_positives"] += 1
    else:                    log["true_negatives"]  += 1

def _write_log(log: dict, model_size: str):
    ts = log["timestamp"].replace(":", "-").replace("T", "_")
    name = f"Model{model_size}_YOLO2SAM_{ts}.json"

    log_dir = LOG_DIR / "yolo_and_sam"   # <<< NEU: eigener Unterordner
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / name, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print("💾 Log gespeichert:", log_dir / name)

# ---------- Hauptlauf mit Lokalisation ----------
def run_yolo2sam(
    model_size: str,
    seq_folder: str | Path,
    labels_folder: str | Path,
    masks_folder: str | Path | None = None,
    ckpt_path: str | None = None,
    cfg_path: str  | None = None,
    max_side: int = 0,
    # IoU-Optionen
    gt_mask_dir: str | Path | None = None,
    iou_thr: float = 0.5,
    min_pixels: int = 0,
    gt_prefix: str = "train_mask_",   # <<< NEU
):
    """
    Optionaler GT-Masken-Vergleich:
      - Wenn gt_mask_dir angegeben und Datei für Bild vorhanden → IoU & Lokalisation.
      - iou_thr: Schwelle für 'lokal korrekt'
      - min_pixels: zu kleine Vorhersagemasken ignorieren (Rauschen)
    """
    t0 = time.time()
    seq_folder    = Path(seq_folder)
    labels_folder = Path(labels_folder)
    masks_folder  = Path(masks_folder) if masks_folder else None
    gt_mask_dir   = Path(gt_mask_dir) if gt_mask_dir else None
    if masks_folder:
        masks_folder.mkdir(parents=True, exist_ok=True)

    run_log = _new_run_dict(
        model_size=model_size, seq_folder=seq_folder,
        labels_folder=labels_folder, masks_folder=masks_folder,
        gt_mask_dir=gt_mask_dir, iou_thr=iou_thr, min_pixels=min_pixels,
        gt_prefix=gt_prefix
    )

    predictor = _make_predictor(model_size, ckpt_path, cfg_path)

    img_paths = sorted(p for p in seq_folder.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    all_frames, all_segments, names = [], {}, []

    for idx, img_p in enumerate(img_paths):
        # 1) Bild laden (optional skalieren)
        im = Image.open(img_p).convert("RGB")
        W0, H0 = im.size
        if max_side and max(W0, H0) > max_side:
            scale = max_side / max(W0, H0)
            im = im.resize((int(W0 * scale), int(H0 * scale)), Image.LANCZOS)
        else:
            scale = 1.0
        W, H = im.size
        frame = np.array(im)

        # 2) YOLO-Labels → Boxen (Pixel)
        txt = labels_folder / f"{img_p.stem}.txt"
        rows = read_yolo_txt(txt)
        boxes_xyxy = []
        for _, x, y, w, h, _ in rows:
            x1, y1, x2, y2 = yolo_xywhn_to_xyxy_pixels(x, y, w, h, W0, H0)
            if scale != 1.0:
                x1 = int(round(x1 * scale)); y1 = int(round(y1 * scale))
                x2 = int(round(x2 * scale)); y2 = int(round(y2 * scale))
            boxes_xyxy.append((x1, y1, x2, y2))

        # 2b) Guard: keine Detections
        if not boxes_xyxy:
            all_frames.append(frame); all_segments[idx] = {}; names.append(img_p.name)
            _update_metrics_multiids(run_log, img_p.name, {}, masks_folder)
            run_log["per_image"][-1].update({"iou_max": None, "loc_ok": False})
            continue

        # 3) Single-Frame + Box-Prompts
        state = _propagate_single_frame(predictor, frame)
        _add_yolo_boxes(predictor, state, frame_index=0, boxes_xyxy=boxes_xyxy)

        segs = {}
        for f_idx, ids, logits in predictor.propagate_in_video(state):
            if f_idx != 0:
                continue
            for i, oid in enumerate(ids):
                # (logits[i]) kann (1,H,W) sein → auf (H,W) drücken
                m = (logits[i] > 0).squeeze().bool().cpu().numpy()
                if m.ndim != 2:
                    m = np.squeeze(m)
                if min_pixels > 0 and m.sum() < min_pixels:
                    continue
                segs[int(oid)] = m

        # 4) Sammeln & binäre Bild-Eval
        all_frames.append(frame)
        all_segments[idx] = segs
        names.append(img_p.name)
        _update_metrics_multiids(run_log, img_p.name, segs, masks_folder)

        # 5) IoU/Lokalisierung gegen GT (falls vorhanden)
        iou_max = None
        loc_ok  = False
        if gt_mask_dir is not None:
            gt = _load_gt_mask_any(gt_mask_dir, img_p.stem, prefix=gt_prefix)  # <<< Prefix!
            if gt is not None:
                # ggf. auf aktuelle Größe skalieren
                if scale != 1.0 or gt.shape != (H0, W0):
                    gt_img = Image.fromarray(gt.astype(np.uint8) * 255)
                    gt_img = gt_img.resize((W, H), Image.NEAREST)
                    gt = (np.array(gt_img) > 0)

                iou_max = max((_mask_iou(m, gt) for m in segs.values()), default=0.0) if segs else 0.0

                # Bild zählt als „lokal korrekt“, wenn GT positiv und IoU >= Schwelle
                exp = GROUND_TRUTH.get(img_p.stem, False)
                if exp:
                    if iou_max >= iou_thr:
                        run_log["loc_true_positives"] += 1
                    else:
                        run_log["loc_false_negatives"] += 1
                    run_log["iou_sum"] += float(iou_max)
                    run_log["iou_count"] += 1
                loc_ok = (iou_max is not None and iou_max >= iou_thr)

        # per-image ergänzen
        run_log["per_image"][-1].update({
            "iou_max": iou_max,
            "loc_ok": bool(loc_ok),
        })
        if segs:
            run_log["pred_with_mask"] += 1

    # Summaries
    run_log["runtime_seconds"] = round(time.time() - t0, 3)
    run_log["mean_iou_gt_pos"] = (run_log["iou_sum"] / run_log["iou_count"]) if run_log["iou_count"] > 0 else None
    gt_pos = sum(1 for r in run_log["per_image"] if r["expected"])
    if gt_pos > 0 and run_log["gt_mask_dir"]:
        run_log["loc_recall"] = run_log["loc_true_positives"] / gt_pos
    else:
        run_log["loc_recall"] = None
    if run_log["pred_with_mask"] > 0 and run_log["gt_mask_dir"]:
        loc_ok_count = sum(1 for r in run_log["per_image"] if r.get("loc_ok"))
        run_log["loc_precision"] = loc_ok_count / run_log["pred_with_mask"]
    else:
        run_log["loc_precision"] = None

    _write_log(run_log, model_size)
    return all_frames, all_segments, names
