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
# SEQUENCE_AND_IMAGE_KEYS = {
#     "IdQx7wJbzls_1SGjJceIqw": ["801688267152464"],
#     "AgKwDT01j4lQOfCcpx3dI5": [
#         "437509901505971", "495230658777622", "694889785039636",
#         "994877221128635", "1158166624986980",
#     ],
#     "Mm0ox3ISOJ4i68cE17CzvG": ["1329064107777798"],
#     "K1-GjltYRoa_Ua67YCMxfg": ["188039513170850"],
#     "y6znehdp01gv1cfqhxlm2s": ["1520816164937345"],
# }
SEQUENCE_AND_IMAGE_KEYS = {
    "LzmRc7EuQBWYgMiQYxa55A": ["318777479617071"],
    "9vc32hajx0nvlm587djfrz": ["468673947552921","465491958081645","219314743337342"],
    "qguobb95xtkik59tpd3fzu": ["334119098140734","195840622387009"],
    "ds2q5f0furwlk4hxtni1n4": ["138968111556284"],
    "hzpj6dghmxiiii9u3333ok": ["473302127088019"],
    "fTBZyU0XzLE8FrC15HvQg6": ["908559296673026"],
    "uYWlqjN7Vo0XSAIGdnZ5Hm" : ["829308255222305"],
    "C3TbHjN0G5PfRwz7rtYsma" : ["3204037906584852"]
    ###
}

FIELDS = ",".join([
    "id","thumb_original_url","width","height","computed_geometry",
    "computed_compass_angle","captured_at","computed_rotation",
    "camera_parameters","computed_altitude","altitude",
])

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# 1.  Configuration – read access_token; enforce output to sequence_to_test_2
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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

# ⟵ Zielordner fest verdrahtet:
OUTPUT_ROOT = pathlib.Path("pipeline/resources/sequence_to_test_2")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def prompt_for_token() -> str:
    return input("🔑  Enter your Mapillary access token: ").strip()

def _save_bytes_and_meta(img_bytes: bytes, meta: dict, out_dir: pathlib.Path) -> None:
    img_id = meta["id"]
    outfile = out_dir / f"{img_id}.jpg"
    metafile = out_dir / f"{img_id}.json"
    outfile.write_bytes(img_bytes)
    metafile.write_text(json.dumps(meta, indent=2))

def download_single_image(image_key: str, token: str) -> None:
    """Download one image + metadata directly into OUTPUT_ROOT (flat)."""
    url = f"https://graph.mapillary.com/{image_key}?access_token={token}&fields={FIELDS}"
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

    try:
        img_bytes = requests.get(meta["thumb_original_url"], timeout=30).content
    except requests.RequestException as exc:
        print(f"❌  Download failed for {meta.get('id')}: {exc}")
        return

    _save_bytes_and_meta(img_bytes, meta, OUTPUT_ROOT)

    lon, lat = meta.get("computed_geometry", {}).get("coordinates", [None, None])
    captured_ms = meta.get("captured_at")
    if isinstance(captured_ms, (int, float)):
        captured = datetime.datetime.fromtimestamp(captured_ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    else:
        captured = str(captured_ms)[:10] if captured_ms else "unknown"
    w, h = meta.get("width", "?"), meta.get("height", "?")
    print(
        f"✅  Saved → { (OUTPUT_ROOT / (str(meta['id']) + '.jpg')).relative_to(pathlib.Path.cwd()) }\n"
        f"   Resolution: {w}×{h}, captured {captured}, "
        f"{lat if lat is not None else '?':.6f} N {lon if lon is not None else '?':.6f} E"
    )

def download_by_image_id(img_id: str, token: str, out_dir: pathlib.Path) -> None:
    """Lädt genau ein Bild + JSON in out_dir (hier: OUTPUT_ROOT)."""
    url = f"https://graph.mapillary.com/{img_id}?access_token={token}&fields={FIELDS}"
    try:
        meta = requests.get(url, timeout=15).json()
    except requests.RequestException as exc:
        print(f"❌  Metadaten-Request fehlgeschlagen für {img_id}: {exc}")
        return
    if "id" not in meta or "thumb_original_url" not in meta:
        print(f"❌  Konnte Metadaten zu {img_id} nicht laden: {meta}")
        return

    outfile = out_dir / f"{img_id}.jpg"
    if outfile.exists():
        print(f"↷  {outfile.name} existiert, überspringe.")
        return

    try:
        img_bytes = requests.get(meta["thumb_original_url"], timeout=30).content
    except requests.RequestException as exc:
        print(f"❌  Download fehlgeschlagen für {img_id}: {exc}")
        return

    _save_bytes_and_meta(img_bytes, meta, out_dir)
    print(f"✅  Gespeichert → {outfile.relative_to(out_dir.parent)}")

def main() -> None:
    token = mcfg.get("access_token", "").strip()
    if not token:
        token = input("🔑  Mapillary Access-Token: ").strip()
    if not token:
        sys.exit("❌  Kein Token – Abbruch.")

    # Dictionary-Pfad → alles direkt nach pipeline/resources/sequence_to_test_2/
    if SEQUENCE_AND_IMAGE_KEYS:
        print(f"📦  Lade ausgewählte Bilder nach {OUTPUT_ROOT} …")
        for seq_id, image_ids in SEQUENCE_AND_IMAGE_KEYS.items():
            for img_id in image_ids:
                download_by_image_id(img_id, token, OUTPUT_ROOT)
        print("🏁  Alle Dictionary-Bilder verarbeitet.")
        return

    # (Fallback: wenn das Dictionary leer ist, kannst du hier optional weitere Modi einbauen.)
    print("ℹ️  SEQUENCE_AND_IMAGE_KEYS ist leer – nichts zu tun.")

if __name__ == "__main__":
    main()
