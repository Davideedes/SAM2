# pipeline/boxes_io.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

def read_yolo_txt(path: str | Path) -> list[list[float]]:
    """
    Liest eine YOLO-Pred-Datei mit Zeilen:
    'cls x y w h [conf]'
    Gibt Liste von [cls, x, y, w, h, conf] (conf optional → -1.0) zurück, Werte float.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = float(parts[0])
        x, y, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else -1.0
        rows.append([cls, x, y, w, h, conf])
    return rows

def yolo_xywhn_to_xyxy_pixels(
    x: float, y: float, w: float, h: float, W: int, H: int
) -> Tuple[int,int,int,int]:
    """
    (x,y,w,h) sind normalisierte Center/Größe (0..1), W/H sind Bildbreite/-höhe in Pixeln.
    Rückgabe: (x1,y1,x2,y2) Pixel-ints, geclipped auf Bildgrenzen.
    """
    cx, cy = x * W, y * H
    bw, bh = w * W, h * H
    x1 = max(0, int(round(cx - bw/2)))
    y1 = max(0, int(round(cy - bh/2)))
    x2 = min(W-1, int(round(cx + bw/2)))
    y2 = min(H-1, int(round(cy + bh/2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2
