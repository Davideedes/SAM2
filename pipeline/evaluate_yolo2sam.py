# pipeline/evaluate_yolo2sam.py
from __future__ import annotations
from pathlib import Path
import time, datetime, json, tempfile, os
import numpy as np
import torch
from PIL import Image

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from .config import load_ground_truth, LOG_DIR
from .vision import save_mask_npz  # wir laden/resize'n hier selber
from .boxes_io import read_yolo_txt, yolo_xywhn_to_xyxy_pixels

GROUND_TRUTH = load_ground_truth()

# ---------- Hilfsfunktionen ----------
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
        "per_image": [], "runtime_seconds": 0.0,
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
    """
    Erzeugt je Box ein neues Objekt (obj_id = 1..K) im gegebenen Frame.
    """
    for j, (x1,y1,x2,y2) in enumerate(boxes_xyxy, start=1):
        # SAM2 erwartet die Box als (1,4) float32; wir geben sie explizit als kw-arg 'box' rein
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_index,
            obj_id=j,
            points=np.empty((0, 2), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            clear_old_points=True,        # wichtig: echtes Bool
            normalize_coords=True,        # Box/Points sind in Pixeln → SAM2 normalisiert intern
            box=box                       # <<< der eigentliche Fix
        )

def _propagate_single_frame(predictor, frame_np: np.ndarray):
    """
    Wir bauen ein 'Video' mit nur 1 Frame, damit die bestehende API gleich bleibt.
    """
    with tempfile.TemporaryDirectory() as tmp:
        Image.fromarray(frame_np).save(Path(tmp)/"00000.jpg")
        state = predictor.init_state(video_path=tmp)
        return state

def _update_metrics_multiids(log:dict, img_name:str, segs:dict, masks_folder:Path|None):
    """
    segs: dict[obj_id -> mask(bool array)]
    Found = mind. eine Nicht-Leer-Maske.
    NPZ-Save: speichert die größte Maske (nach Pixelanzahl), falls expected=True.
    """
    stem = Path(img_name).stem
    exp  = GROUND_TRUTH.get(stem, False)
    areas = {oid:int(m.sum()) for oid,m in segs.items()} if segs else {}
    found = any(a > 0 for a in areas.values())

    if exp and found and masks_folder and areas:
        best_id = max(areas, key=areas.get)
        save_mask_npz(segs[best_id], masks_folder/f"{stem}.npz")

    log["per_image"].append({"img_name":img_name,"expected":exp,"found":found,
                             "n_objects": len(segs),
                             "type":("tp" if exp and found else
                                     "tn" if not exp and not found else
                                     "fn" if exp and not found else "fp")})
    if exp:   log["expected_total"] += 1
    if found: log["found_total"]    += 1
    if   exp and found:      log["true_positives"]  += 1
    elif exp and not found:  log["false_negatives"] += 1
    elif not exp and found:  log["false_positives"] += 1
    else:                    log["true_negatives"]  += 1

def _write_log(log:dict, model_size:str):
    ts = log["timestamp"].replace(":","-").replace("T","_")
    name = f"Model{model_size}_YOLO2SAM_{ts}.json"
    with open(LOG_DIR/name,"w",encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print("💾 Log gespeichert:", LOG_DIR/name)

# ---------- Hauptlauf ----------
def run_yolo2sam(
    model_size:str,
    seq_folder:str | Path,
    labels_folder:str | Path,
    masks_folder:str | Path | None = None,
    ckpt_path:str | None = None,
    cfg_path:str  | None = None,
    max_side:int = 0,
):
    """
    Läuft über alle Bilder in seq_folder, liest pro Bild die YOLO-Labels (labels_folder),
    baut Box-Prompts und segmentiert mit SAM2. Optional Masken speichern & Metriken loggen.

    max_side: 0 = Originalauflösung verwenden; >0 = längste Bildseite auf diesen Wert skalieren.
              Die Box-Koordinaten werden entsprechend mitskaliert.
    """
    t0 = time.time()
    seq_folder   = Path(seq_folder)
    labels_folder= Path(labels_folder)
    masks_folder = Path(masks_folder) if masks_folder else None
    if masks_folder:
        masks_folder.mkdir(parents=True, exist_ok=True)

    run_log = _new_run_dict(model_size=model_size, seq_folder=seq_folder,
                            labels_folder=labels_folder, masks_folder=masks_folder)

    predictor = _make_predictor(model_size, ckpt_path, cfg_path)

    img_paths = sorted(p for p in seq_folder.iterdir()
                       if p.suffix.lower() in {".jpg",".jpeg",".png"})
    all_frames, all_segments, names = [], {}, []

    for idx, img_p in enumerate(img_paths):
        # 1) Bild laden (optional skalieren)
        im = Image.open(img_p).convert("RGB")
        W0, H0 = im.size
        if max_side and max(W0,H0) > max_side:
            scale = max_side / max(W0,H0)
            im = im.resize((int(W0*scale), int(H0*scale)), Image.LANCZOS)
        else:
            scale = 1.0
        W, H = im.size
        frame = np.array(im)

        # 2) YOLO-Labels einlesen & in (x1,y1,x2,y2) Pixel (auf _aktuelle_ Größe) umrechnen
        txt = labels_folder / f"{img_p.stem}.txt"
        rows = read_yolo_txt(txt)  # jede Zeile: [cls, x, y, w, h, conf]
        boxes_xyxy = []
        for _, x, y, w, h, _ in rows:
            # rows sind auf Originalbild normalisiert → zuerst in Original-Pixel, dann skalieren
            x1, y1, x2, y2 = yolo_xywhn_to_xyxy_pixels(x, y, w, h, W0, H0)
            if scale != 1.0:
                x1 = int(round(x1 * scale)); y1 = int(round(y1 * scale))
                x2 = int(round(x2 * scale)); y2 = int(round(y2 * scale))
            boxes_xyxy.append((x1,y1,x2,y2))
            
        txt = labels_folder / f"{img_p.stem}.txt"
        rows = read_yolo_txt(txt)
        boxes_xyxy = []
        for _, x, y, w, h, _ in rows:
            x1, y1, x2, y2 = yolo_xywhn_to_xyxy_pixels(x, y, w, h, W0, H0)
            if scale != 1.0:
                x1 = int(round(x1 * scale)); y1 = int(round(y1 * scale))
                x2 = int(round(x2 * scale)); y2 = int(round(y2 * scale))
            boxes_xyxy.append((x1, y1, x2, y2))

        # 👉 Guard: keine Detections → SAM2 überspringen
        if not boxes_xyxy:
            all_frames.append(frame)
            all_segments[idx] = {}
            names.append(img_p.name)
            _update_metrics_multiids(run_log, img_p.name, {}, masks_folder)
            continue

        # 3) Single-Frame State, Box-Prompts hinzufügen, propagieren
        state = _propagate_single_frame(predictor, frame)
        _add_yolo_boxes(predictor, state, frame_index=0, boxes_xyxy=boxes_xyxy)

        segs = {}
        for f_idx, ids, logits in predictor.propagate_in_video(state):
            if f_idx != 0:
                continue
            segs = {oid: (logits[i] > 0).cpu().numpy().astype(np.bool_) for i, oid in enumerate(ids)}

        # 4) Sammeln & Auswertung
        all_frames.append(frame)
        all_segments[idx] = segs
        names.append(img_p.name)

        _update_metrics_multiids(run_log, img_p.name, segs, masks_folder)

    run_log["runtime_seconds"] = round(time.time() - t0, 3)
    _write_log(run_log, model_size)
    return all_frames, all_segments, names
