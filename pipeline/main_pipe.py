"""
main_pipeline.py
~~~~~~~~~~~~~~~~
End-to-End-Ablauf:
1. Interaktive SAM-2-Segmentierung
2. Masken- & BBox-Overlay speichern
3. Florence-2-Caption erzeugen
"""

# Aufrufbeispiel:
#python3 pipeline/main_pipe.py --image testbilder/CLXQ7779.JPG --outdir generated_masks
# -----------------------------------------------------------

import sys, pathlib, logging, time
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from pathlib import Path
import argparse
from sam2_module import Sam2Segmenter
from florence2_module import Florence2Classifier
from PIL import Image

# ────────────────────────── Logging ──────────────────────────
def _init_logger(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log_file = outdir / "pipeline.log"
    fmt = "%(asctime)s  [%(levelname)s]  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, encoding="utf-8")],
    )
    return logging.getLogger("Pipeline")

# ────────────────────────── Pipeline ─────────────────────────
def run(
    image_path: str,
    outdir: str = ".",
    model_sam: str = "facebook/sam2-hiera-tiny",
):
    t0          = time.time()
    outdir_path = Path(outdir)
    log         = _init_logger(outdir_path)

    log.info("Starte Pipeline")
    log.info(f"Eingabebild: {image_path}")

    # ---------- 1) SAM-2 Interaktion ---------------------------------
    sam = Sam2Segmenter(model_name=model_sam)
    log.info("SAM-2 Modell geladen")

    mask_bool, image_np = sam.segment_interactive(image_path)
    log.info("Interaktive Segmentierung abgeschlossen")

    mask_path = outdir_path / "mask.png"
    sam.save_mask(mask_bool, mask_path)
    log.info(f"Binärmaske gespeichert → {mask_path}")

    masked_img = Sam2Segmenter.apply_mask_with_context(image_np, mask_bool)
    (outdir_path / "masked_object.png").write_bytes(masked_img.tobytes())

    # ---------- BBox + Crop ------------------------------------------
    bbox            = sam.mask_to_bbox(mask_bool, padding=0.10)
    orig_img_pil    = Image.fromarray(image_np)
    bbox_overlay    = sam.draw_bbox_on_image(orig_img_pil, bbox)
    bbox_overlay.save(outdir_path / "bbox_overlay.png")
    cropped_img     = sam.crop_bbox(image_np, bbox)
    cropped_img.save(outdir_path / "cropped_object.png")

    # … unverändert bis zum Florence-Teil …

    # ---------- 2) Florence-2 Analyse --------------------------------
    florence = Florence2Classifier()
    log.info("Florence-2 Modell geladen – prüfe Pothole + Caption")

    # ---- 2a) Pothole-Check  (NEU) -----------------------------------
    has_pothole, detections = florence.detect_pothole(orig_img_pil)
    if has_pothole:
        log.info(f"POTHOLE DETECTED  →  {len(detections)} Treffer")
    else:
        log.info("Kein Schlagloch gefunden")

    # optional: BBox-Overlay für bestes Ergebnis abspeichern
    if detections:
        best_box, best_score = max(detections, key=lambda x: x[1])
        pothole_overlay = sam.draw_bbox_on_image(orig_img_pil, best_box, color="yellow")
        pothole_overlay.save(outdir_path / "pothole_overlay.png")

    # ---- 2b) Caption  (wie gehabt) ----------------------------------
    try:
        caption = florence.classify(
            image       = cropped_img,
            task_prompt = "<DETAILED_CAPTION>",
            text_input  = None,
        )
        log.info(f"Caption: {caption}")
    except Exception:
        log.exception("Fehler bei Florence-2")
        caption = "<FEHLER>"

# ---------- Abschluss-Print -------------------------------------------------
    print("\n🖼  Florence-2 Caption :", caption)
    print("🚧 Pothole erkannt   :", has_pothole)
    if has_pothole:
        print("BBox-Overlay        :", outdir_path / 'pothole_overlay.png')
    print("Maske               :", mask_path)
    print("Mask-Overlay        :", outdir_path / 'masked_object.png')


# ────────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SAM2 → Florence-2 Pipeline")
    p.add_argument("--image",     required=True, help="Pfad zum Eingabebild")
    p.add_argument("--outdir",    default="output", help="Ausgabe-Verzeichnis")
    p.add_argument("--sam_model", default="facebook/sam2-hiera-tiny",
                   help="HF-ID oder lokaler Pfad des SAM-2-Modells")
    args = p.parse_args()

    run(args.image, args.outdir, args.sam_model)
