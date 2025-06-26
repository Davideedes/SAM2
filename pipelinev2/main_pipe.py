"""
Haupt-Entry:  SAM-2  ➜  Mask-Crops  ➜  Florence-2-Analyse
"""

import sys, pathlib, logging, time
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))


from sam2_flow   import run_sam2_pipeline
from florence2_flow import run_florence_on_folder

IMG_FOLDER   = "./testbilder"          # hier liegen Schlagloch-Bilder
CROP_FOLDER  = "./testbilder/masken_ctx"

if __name__ == "__main__":
    # 1) SAM-2: Masken erzeugen + Crops abspeichern
    run_sam2_pipeline(IMG_FOLDER, CROP_FOLDER, n_click_frames=4)

    # 2) Florence-2: alle Crops auswerten
    run_florence_on_folder(CROP_FOLDER)
