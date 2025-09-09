# pipeline/run_mapillary_yolo_hamburg.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

import configparser
import requests
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------- Pfade & Konfiguration -----------------
# Datei liegt in pipeline/, also ist Repo-Root = parents[1]
ROOT = Path(__file__).resolve().parents[2]

# === WEIGHTS: robust suchen (bevorzugt yolov12/best.pt) ===
WEIGHTS_CANDIDATES = [
    ROOT / "yolov12/pothole_yolov12_google_colab_training/weights/best.pt"
]
WEIGHTS = next((p for p in WEIGHTS_CANDIDATES if p.exists()), WEIGHTS_CANDIDATES[0])

# === Output-Ziele: alles unter pipeline/test ===
PROJECT = ROOT / "pipeline/test/yolo_runs"
OUT_POSITIVES = ROOT / "pipeline/test/positives"
PROJECT.mkdir(parents=True, exist_ok=True)
OUT_POSITIVES.mkdir(parents=True, exist_ok=True)

RUN_NAME = f"pred_hh_{time.strftime('%Y%m%d_%H%M%S')}_i1024_c0.2_u0.6_stream"

IMG_SIZE  = 1024
CONF      = 0.20
IOU       = 0.60
DEVICE    = 0        # CUDA: 0, CPU: 'cpu'
HALF      = True

# Trefferlimit: nach N positiven Bildern abbrechen
POSITIVE_LIMIT = 20

# ----------------- Mapillary Auth (Token) -----------------
def load_access_token() -> str:
    cfg = configparser.ConfigParser()
    # übliche Orte probieren (in dieser Reihenfolge)
    candidates = [
        ROOT / "pipeline/test/fetch_mpy_images.ini",
        ROOT / "fetch_mpy_images.ini",
        Path(__file__).with_suffix(".ini"),
        ROOT / "yolov12/fetch_mpy_images.ini",
    ]
    for c in candidates:
        if c.exists():
            cfg.read(c)
            if cfg.has_section("mapillary") and cfg.has_option("mapillary", "access_token"):
                return cfg.get("mapillary", "access_token").strip()
    return ""

ACCESS_TOKEN = load_access_token()
if not ACCESS_TOKEN:
    ACCESS_TOKEN = input("🔑  Mapillary Access-Token: ").strip()
if not ACCESS_TOKEN:
    raise SystemExit("❌  Kein Mapillary-Token – Abbruch.")

# Felder, die wir für jeden Treffer brauchen
FIELDS = ",".join([
    "id","thumb_original_url","width","height","computed_geometry",
    "captured_at","computed_compass_angle","computed_rotation"
])

# ----------------- Hamburg-Gebiet -----------------
# Grobe BBox für Hamburg (WGS84): minLon, minLat, maxLon, maxLat
HAMBURG_BBOX = (9.60, 53.30, 10.30, 53.75)

# Optional: BBox in Kacheln splitten, um Timeouts zu vermeiden
GRID_DEG = 0.10  # ~11 km N-S

# ---------------------------------------------------
def tile_bbox(bbox, step_deg=0.1) -> Iterator[tuple[float,float,float,float]]:
    minx, miny, maxx, maxy = bbox
    x = minx
    while x < maxx - 1e-12:
        nx = min(x + step_deg, maxx)
        y = miny
        while y < maxy - 1e-12:
            ny = min(y + step_deg, maxy)
            yield (x, y, nx, ny)
            y = ny
        x = nx

def decode_to_numpy(img_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def has_pothole(result) -> bool:
    # Wenn dein Modell nur eine Klasse hat, reicht >0 Boxes.
    return result.boxes is not None and len(result.boxes) > 0

def save_positive(meta: Dict[str,Any], img_bytes: bytes) -> None:
    img_id = meta["id"]
    (OUT_POSITIVES / f"{img_id}.jpg").write_bytes(img_bytes)
    (OUT_POSITIVES / f"{img_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def fetch_image_meta_page(bbox: tuple[float,float,float,float], limit=200, cursor: Optional[str]=None) -> dict:
    params = {
        "access_token": ACCESS_TOKEN,
        "bbox": ",".join(map(str, bbox)),
        "fields": FIELDS,
        "limit": str(limit),
    }
    if cursor:
        params["after"] = cursor
    url = "https://graph.mapillary.com/images"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def stream_images_in_bbox(bbox: tuple[float,float,float,float]) -> Iterator[Dict[str,Any]]:
    cursor = None
    while True:
        data = fetch_image_meta_page(bbox, limit=200, cursor=cursor)
        for item in data.get("data", []):
            if "thumb_original_url" not in item:
                # Nachladen der Felder für die ID (manche Antworten lassen Felder weg)
                u = f"https://graph.mapillary.com/{item['id']}"
                r = requests.get(u, params={"access_token": ACCESS_TOKEN, "fields": FIELDS}, timeout=15)
                if r.ok:
                    item = r.json()
            yield item

        paging = data.get("paging", {})
        cursors = paging.get("cursors", {})
        after = cursors.get("after")
        if after:
            cursor = after
            continue
        next_url = paging.get("next")
        if next_url:
            r = requests.get(next_url, timeout=30)
            if r.ok:
                data = r.json()
                for item in data.get("data", []):
                    yield item
                paging = data.get("paging", {})
                cursors = paging.get("cursors", {})
                cursor = cursors.get("after")
                if cursor:
                    continue
        break

def download_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except requests.RequestException:
        return None

def main():
    print("== Streaming YOLO run (Hamburg) ==")
    print("repo root:", ROOT)
    print("weights  :", WEIGHTS, WEIGHTS.exists())
    if not WEIGHTS.exists():
        print("⚠️  Konnte best.pt nicht finden. Probierte Kandidaten:")
        for p in WEIGHTS_CANDIDATES:
            print("   -", p, "(exists:", p.exists(), ")")
        raise SystemExit("❌  weights/best.pt nicht gefunden – bitte WEIGHTS anpassen.")

    print("project  :", PROJECT)
    print("run      :", RUN_NAME)

    model = YOLO(str(WEIGHTS))

    positives = 0
    processed = 0

    for bbox in tile_bbox(HAMBURG_BBOX, step_deg=GRID_DEG):
        print(f"→ Suche in Tile BBox={bbox}")
        for meta in stream_images_in_bbox(bbox):
            processed += 1
            if "id" not in meta or "thumb_original_url" not in meta:
                continue

            img_bytes = download_bytes(meta["thumb_original_url"])
            if not img_bytes:
                # temporäre URL ggf. erneuern
                u = f"https://graph.mapillary.com/{meta['id']}"
                r = requests.get(u, params={"access_token": ACCESS_TOKEN, "fields": FIELDS}, timeout=15)
                if r.ok and "thumb_original_url" in r.json():
                    img_bytes = download_bytes(r.json()["thumb_original_url"])
                if not img_bytes:
                    print(f"✗ Bild nicht erreichbar: {meta['id']}")
                    continue

            np_img = decode_to_numpy(img_bytes)
            if np_img is None:
                print(f"✗ Decoding fehlgeschlagen: {meta['id']}")
                continue

            # Erst “trocken” laufen lassen (nichts speichern)
            res = model.predict(
                source=np_img,
                imgsz=IMG_SIZE,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                half=HALF,
                save=False,
                save_txt=False,
                save_conf=False,
                verbose=False
            )
            r0 = res[0]

            if has_pothole(r0):
                # YOLO mit Speichern erneut laufen lassen (legt Annotierungen/Labels ab)
                model.predict(
                    source=np_img,
                    imgsz=IMG_SIZE,
                    conf=CONF,
                    iou=IOU,
                    device=DEVICE,
                    half=HALF,
                    save=True,
                    save_txt=True,
                    save_conf=True,
                    project=str(PROJECT),
                    name=RUN_NAME,
                    exist_ok=True,
                    verbose=False
                )
                # Original + JSON nur bei POSITIV sichern
                save_positive(meta, img_bytes)

                positives += 1
                print(f"✅ Treffer {positives}/{POSITIVE_LIMIT} – id={meta['id']}")
                if positives >= POSITIVE_LIMIT:
                    print("🏁 Limit erreicht – Abbruch.")
                    print("LABELS_DIR:", PROJECT / RUN_NAME / "labels")
                    return

            # Gentle pacing (Rate-Limits)
            time.sleep(0.03)

    print(f"Fertig. Verarbeitet: {processed}, Treffer: {positives}")
    print("LABELS_DIR:", PROJECT / RUN_NAME / "labels")

if __name__ == "__main__":
    main()
