# yolov12/run_yolo.py
from ultralytics import YOLO
from pathlib import Path
import time

# Fixe Pfade relativ zum Repo-Root (Script liegt im Ordner 'yolov12')
ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = ROOT / "yolov12/pothole_yolov12_google_colab_training/weights/best.pt"
SOURCE  = ROOT / "pipeline/resources/sequence_to_test_2"
PROJECT = ROOT / "pipeline/resources/yolo_runs"
NAME    = f"pred_{time.strftime('%Y%m%d_%H%M%S')}_i1024_c0.2_u0.6"

print("== YOLO run ==")
print("cwd     :", Path().resolve())
print("weights :", WEIGHTS, WEIGHTS.exists())
print("source  :", SOURCE,  SOURCE.exists())
print("project :", PROJECT)

model = YOLO(str(WEIGHTS))

results = model.predict(
    source=str(SOURCE),
    imgsz=1024,
    conf=0.70,
    iou=0.60,
    device=0,          # WSL+CUDA
    half=True,
    save=True,
    save_txt=True,
    save_conf=True,
    project=str(PROJECT),
    name=NAME,
    exist_ok=True,
    verbose=False
)

labels_dir = PROJECT / NAME / "labels"
print("LABELS_DIR:", labels_dir)
