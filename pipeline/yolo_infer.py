# pipeline/yolo_infer.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
import numpy as np

def run_yolo_folder(
    weights: str,
    source_dir: str | Path,
    labels_out: str | Path,
    imgsz: int = 1024,
    conf: float = 0.20,
    iou: float = 0.60,
    device: int | str = 0,
    classes: Optional[Iterable[int]] = None,
    overwrite: bool = True,
) -> None:
    """
    Führt YOLO-Inferenz über alle Bilder in source_dir aus und schreibt pro Bild
    eine .txt-Datei im YOLO-Pred-Format: 'cls x y w h conf' (x/y/w/h: normalisiert 0..1).
    """
    from ultralytics import YOLO

    source_dir = Path(source_dir)
    labels_out = Path(labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    results = model.predict(
        source=str(source_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,   # 0 == "cuda:0" in WSL, CPU: "cpu"
        stream=True,     # wir iterieren über Ergebnisse und speichern selbst
        verbose=False,
        save=False, save_txt=False
    )

    for r in results:
        stem = Path(r.path).stem
        out_txt = labels_out / f"{stem}.txt"
        if out_txt.exists() and not overwrite:
            continue

        if r.boxes is None or r.boxes.shape[0] == 0:
            out_txt.write_text("")  # leere Datei → keine Detections
            continue

        xywhn = r.boxes.xywhn.cpu().numpy()   # (N,4), normierte Werte
        cls   = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        confs = r.boxes.conf.cpu().numpy()    # (N,)

        lines = []
        for c, (x, y, w, h), p in zip(cls, xywhn, confs):
            if classes is not None and c not in classes:
                continue
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {p:.4f}")

        out_txt.write_text("\n".join(lines))
