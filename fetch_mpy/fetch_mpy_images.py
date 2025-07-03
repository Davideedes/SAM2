from __future__ import annotations
import pathlib
import sys
import requests
import datetime
import json
import shutil
import configparser


# ────────────────────────────────────────────────────────────────────────────────
# 1.  Configuration – values come from config.ini but fall back to sane defaults
# ────────────────────────────────────────────────────────────────────────────────
 # Look for an INI file named either fetch_mpy_images.ini (same basename) or a generic config.ini
candidate1 = pathlib.Path(__file__).with_name("fetch_mpy_images.ini")
candidate2 = pathlib.Path(__file__).with_suffix(".ini")        
CONFIG_FILE = candidate1 if candidate1.exists() else candidate2
if not CONFIG_FILE.exists():
    print("⚠️  No config.ini / fetch_mpy_images.ini found – using built‑in defaults.")
cfg = configparser.ConfigParser()
cfg.read(CONFIG_FILE)
mcfg = cfg["mapillary"] if "mapillary" in cfg else {}

# Handle an optional bounding‑box string "minLon,minLat,maxLon,maxLat"
_bbox_raw = mcfg.get("bbox", "").replace(" ", "")
if _bbox_raw:
    try:
        BBOX = tuple(map(float, _bbox_raw.split(",")))
        if len(BBOX) != 4:
            raise ValueError
    except ValueError:
        sys.exit("❌  The bbox entry in config.ini must be four comma‑separated numbers.")
else:
    BBOX = None  # user chose to leave it blank

# Optional keys to override the BBOX query
SEQUENCE_KEY = mcfg.get("sequence_key", "")
IMAGE_KEY    = mcfg.get("image_key", "")

# Number of images to download for a BBOX query
LIMIT        = int(mcfg.get("limit", 1))

# Root folder under which download sub‑folders are created
OUTPUT_ROOT  = pathlib.Path(mcfg.get("output_root", "downloads"))

# Mapillary Graph API fields we always request
FIELDS = ",".join([
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
    "altitude"
])

# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    token = mcfg.get("access_token", "").strip()
    if not token:
        token = input("🔑  Enter your Mapillary access token: ").strip()
    # ────────────────────────────────────────────────────────────────────────
    # Derive an output directory that reflects the query type:
    #   • image  →  image_<IMAGE_KEY>
    #   • sequence → sequence_<SEQUENCE_KEY>
    #   • bbox   →  bbox_<minLon>_<minLat>_<maxLon>_<maxLat>
    # ────────────────────────────────────────────────────────────────────────
    if IMAGE_KEY:
        base_dir = OUTPUT_ROOT / f"image_{IMAGE_KEY}"
    elif SEQUENCE_KEY:
        base_dir = OUTPUT_ROOT / f"sequence_{SEQUENCE_KEY}"
    else:
        if BBOX is None:
            sys.exit("❌  No bbox configured and no image/sequence key provided.")
        bbox_str = "_".join(map(str, BBOX))
        base_dir = OUTPUT_ROOT / f"bbox_{bbox_str}"
    base_dir.mkdir(parents=True, exist_ok=True)

    if not token:
        sys.exit("❌  No token provided. Exiting.")

    # decide how to query: by image id, by sequence id, or by bbox
    if IMAGE_KEY:
        print("📡  Querying Mapillary …")
        url = (
            f"https://graph.mapillary.com/{IMAGE_KEY}"
            f"?access_token={token}"
            f"&fields={FIELDS}"
        )
        resp = requests.get(url, timeout=15).json()
        meta = resp  # single-image endpoint returns a dict, not a list
        img_id = meta["id"]
        OUTFILE = base_dir / f"{img_id}.jpg"
        METAFILE = OUTFILE.with_suffix(".json")
    elif SEQUENCE_KEY:
        print("📡  Querying Mapillary for the whole sequence …")
        OUTPUT_DIR = base_dir
        meta_list = []
        next_url = (
            f"https://graph.mapillary.com/images"
            f"?access_token={token}"
            f"&fields={FIELDS}"
            f"&sequence_ids={SEQUENCE_KEY}"
            f"&limit=500"
        )
        while next_url:
            resp = requests.get(next_url, timeout=15).json()
            meta_list.extend(resp.get("data", []))
            next_url = resp.get("paging", {}).get("next")
        if not meta_list:
            sys.exit("❌  No imagery found for the given sequence key")
        print(f"⚙️  Found {len(meta_list)} images; downloading …")
        for meta in meta_list:
            img_url   = meta["thumb_original_url"]
            img_id    = meta["id"]
            img_bytes = requests.get(img_url, timeout=30).content
            (OUTPUT_DIR / f"{img_id}.jpg").write_bytes(img_bytes)
            (OUTPUT_DIR / f"{img_id}.json").write_text(json.dumps(meta, indent=2))
        print(f"✅  Saved {len(meta_list)} images + metadata in {OUTPUT_DIR}")
        return
    else:
        if BBOX is None:
            sys.exit("❌  No bbox configured in config.ini (bbox=) and no other query parameters provided.")
        bbox_param = ",".join(map(str, BBOX))
        print("📡  Querying Mapillary …")
        url = (
            f"https://graph.mapillary.com/images"
            f"?access_token={token}"
            f"&fields={FIELDS}"
            f"&bbox={bbox_param}"
            f"&limit={LIMIT}"
        )
        resp = requests.get(url, timeout=15).json()
        data = resp.get("data", [])
        if not data:
            sys.exit("❌  No imagery found in the specified bounding box")
        meta = data[0]
        img_id = meta["id"]
        OUTFILE = base_dir / f"{img_id}.jpg"
        METAFILE = OUTFILE.with_suffix(".json")

    img_url = meta["thumb_original_url"]
    print(f"⇩  Downloading {img_url}")
    img_bytes = requests.get(img_url, timeout=30).content
    OUTFILE.write_bytes(img_bytes)

    # save the full metadata alongside the image
    METAFILE.write_text(json.dumps(meta, indent=2))

    lon, lat = meta["computed_geometry"]["coordinates"]
    captured_ms = meta.get("captured_at")
    if isinstance(captured_ms, (int, float)):
        captured = datetime.datetime.fromtimestamp(captured_ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    else:
        captured = str(captured_ms)[:10] if captured_ms else "unknown"
    w, h = meta["width"], meta["height"]

    print(
        f"✅  Saved → {OUTFILE} and {METAFILE}\n"
        f"   Resolution: {w}×{h}, captured {captured}, {lat:.6f} N {lon:.6f} E"
    )


if __name__ == "__main__":
    main()