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



"""
Haupt-Entry:  SAM-2  ➜  Mask-Crops  ➜  Florence-2-Analyse
"""
from sam2_module   import run_sam2_pipeline
from florence2_module import run_florence_on_folder

IMG_FOLDER   = "./testbilder"          # hier liegen Schlagloch-Bilder
CROP_FOLDER  = "./testbilder/masken_ctx"

if __name__ == "__main__":
    # 1) SAM-2: Masken erzeugen + Crops abspeichern
    run_sam2_pipeline(IMG_FOLDER, CROP_FOLDER, n_click_frames=4)

    # 2) Florence-2: alle Crops auswerten
    run_florence_on_folder(CROP_FOLDER)
