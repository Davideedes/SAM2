from __future__ import annotations

import datetime
import json
import pathlib
import sys
import time
from typing import Dict, List

import configparser
import requests

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# 0.  Define the set of images to fetch (sequence → [image IDs])
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
SEQUENCE_AND_IMAGE_KEYS= {
    "y0b917m09znglhifax9sqv": [
        "411774873172944",
        "1358223377911569",
        "2573485702947530",
        "811905316370016",
    ],
    "YLl6SUNdgjAH09OVJPbo82": [
        "1132980251464779",
        "910210667858716",
    ],
    "zrOQJgMsECZxKa6cPNnW53": [
        "222756804264436",
        "2133396897024110",
        "1634974493981268",
        "482100944477332",
        "342417245330517",
        "3829640733992418",
        "25664145879899175",
        "1675542486537246",
        "1528844847843726",
    ],
    "lbahjpjck702xu6kha7ugc": [
        "1173754136384887",
    ],
    "lqw0j1s11kbsrqrjl263dx": [
        "134501391962606",
        "293952292211136",
        "284309256666805",
    ],
}

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# 1.  Configuration – read access_token/output_root from INI (if present)
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
FIELDS = ",".join(
    [
        "id",
        "thumb_original_url",
        "width",
        "height",
        "computed_geometry",
        "computed_compass_angle",
        "captured_at",
        "computed_rotation",
        "camera_parameters",
        "computed_altitude",
        "altitude",
    ]
)

# Locate ini (same logic as original script: <name>.ini or fetch_mpy_images.ini)
self_path = pathlib.Path(__file__)
candidate1 = self_path.with_name("fetch_mpy_images.ini")
candidate2 = self_path.with_suffix(".ini")
CONFIG_FILE = candidate1 if candidate1.exists() else candidate2
cfg = configparser.ConfigParser()
if CONFIG_FILE.exists():
    cfg.read(CONFIG_FILE)
    mcfg = cfg["mapillary"] if cfg.has_section("mapillary") else {}

else:
    print("⚠️  No config.ini / fetch_mpy_images.ini found – will ask for token at runtime.")
    mcfg = {}

access_token = mcfg.get("access_token", "").strip()
OUTPUT_ROOT = pathlib.Path(mcfg.get("output_root", "downloads"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def prompt_for_token() -> str:
    """Prompt once for the Mapillary token (stdin) and return it."""
    return input("🔑  Enter your Mapillary access token: ").strip()


def download_single_image(image_key: str, token: str) -> None:
    """Download one image + metadata given the image_key."""
    url = (
        f"https://graph.mapillary.com/{image_key}"
        f"?access_token={token}"
        f"&fields={FIELDS}"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        meta = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"❌  Failed to fetch metadata for {image_key}: {exc}")
        return

    if "id" not in meta or "thumb_original_url" not in meta:
        print(f"❌  No usable data returned for {image_key} – skipping.")
        return

    img_id = meta["id"]
    img_url = meta["thumb_original_url"]

    # Each image gets its own sub‑folder (image_<ID>) – mirrors original behavior
    out_dir = OUTPUT_ROOT / f"image_{img_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    outfile = out_dir / f"{img_id}.jpg"
    metafile = outfile.with_suffix(".json")

    try:
        img_bytes = requests.get(img_url, timeout=30).content
    except requests.RequestException as exc:
        print(f"❌  Download failed for {img_id}: {exc}")
        return

    outfile.write_bytes(img_bytes)
    metafile.write_text(json.dumps(meta, indent=2))

    # Console summary like the original script
    lon, lat = meta.get("computed_geometry", {}).get("coordinates", [None, None])
    captured_ms = meta.get("captured_at")
    if isinstance(captured_ms, (int, float)):
        captured = datetime.datetime.fromtimestamp(captured_ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    else:
        captured = str(captured_ms)[:10] if captured_ms else "unknown"

    w, h = meta.get("width", "?"), meta.get("height", "?")
    print(
        f"✅  Saved → {outfile.relative_to(pathlib.Path.cwd())}\n"
        f"   Resolution: {w}×{h}, captured {captured}, "
        f"{lat if lat is not None else '?':.6f} N {lon if lon is not None else '?':.6f} E"
    )
def download_by_image_id(img_id: str, token: str, out_dir: pathlib.Path) -> None:
    """Lädt genau ein Bild + JSON in out_dir – wird von der Dictionary-Schleife aufgerufen."""
    url = (
        f"https://graph.mapillary.com/{img_id}"
        f"?access_token={token}"
        f"&fields={FIELDS}"
    )
    meta = requests.get(url, timeout=15).json()
    if "id" not in meta:
        print(f"❌  Konnte Metadaten zu {img_id} nicht laden: {meta}")
        return

    img_url = meta["thumb_original_url"]
    OUTFILE  = out_dir / f"{img_id}.jpg"
    METAFILE = OUTFILE.with_suffix(".json")

    if OUTFILE.exists():          # Skip, falls schon vorhanden
        print(f"↷  {OUTFILE.name} existiert, überspringe.")
        return

    img_bytes = requests.get(img_url, timeout=30).content
    OUTFILE.write_bytes(img_bytes)
    METAFILE.write_text(json.dumps(meta, indent=2))
    print(f"✅  Gespeichert → {OUTFILE.relative_to(out_dir.parent)}")


def main() -> None:
    # … unverändert bis zum Zugriffstoken:
    token = mcfg.get("access_token", "").strip()
    if not token:
        token = input("🔑  Mapillary Access-Token: ").strip()
    if not token:
        sys.exit("❌  Kein Token – Abbruch.")

    # ───────────────────────────────────────────────────────────────────────
    #  NEUER DICTIONARY-PFAD: nur ausgewählte Bilder herunterladen
    # ───────────────────────────────────────────────────────────────────────
    if SEQUENCE_AND_IMAGE_KEYS:          # nur wenn nicht leer
        print("📦  Starte Batch-Download für ausgewählte Bilder …")
        for seq_id, image_ids in SEQUENCE_AND_IMAGE_KEYS.items():
            seq_dir = OUTPUT_ROOT / f"sequence_{seq_id}"
            seq_dir.mkdir(parents=True, exist_ok=True)
            for img_id in image_ids:
                download_by_image_id(img_id, token, seq_dir)
        print("🏁  Alle Dictionary-Bilder verarbeitet.")
        return
    # ───────────────────────────────────────────────────────────────────────
    #  Falls Dictionary leer → alter Codepfad (image_key / sequence_key / bbox)
    # ───────────────────────────────────────────────────────────────────────
    # (hier folgt dein bestehender Code unverändert …)


if __name__ == "__main__":
    main()
