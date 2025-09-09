#!/usr/bin/env bash
set -euo pipefail

# ================= USER CONFIG ======================
SEQ_FOLDER="pipeline/resources/sequence_to_test_2"
MASKS_OUT="pipeline/resources/generated_npz_masks_yolo_only/y8nseg"

WEIGHTS="yolov12/yolov12n-seg.pt"   # oder eigener Pfad zu deinem Seg-Checkpoint
IMG_SIZE=1024
CONF=0.20
NMS_IOU=0.60
DEVICE=0                  # "cpu" für CPU

CLASSES=""                # z.B. "0,2,5" (leer = alle)
MIN_PIXELS=1000
SAVE_PNG=false            # true → Instanz-PNGs unter masks_out/instances_png/

MERGE="largest"           # largest | sum | none

# ---- GT-Masken / Eval ----
GT_MASK_DIR="pipeline/resources/sequence_to_test_2_npz_masks_ground_truth"
GT_PREFIX="train_mask_"
IOU_THR=0.3
# ====================================================

mkdir -p "${MASKS_OUT}"

python -m pipeline.cli_yolo_only \
  --weights "${WEIGHTS}" \
  --seq-folder "${SEQ_FOLDER}" \
  --masks-folder "${MASKS_OUT}" \
  --imgsz "${IMG_SIZE}" \
  --conf "${CONF}" \
  --iou "${NMS_IOU}" \
  --device "${DEVICE}" \
  ${CLASSES:+--classes "${CLASSES}"} \
  --min-pixels "${MIN_PIXELS}" \
  --merge "${MERGE}" \
  ${SAVE_PNG:+--save-png-instances} \
  ${GT_MASK_DIR:+--gt-mask-dir "${GT_MASK_DIR}"} \
  --gt-prefix "${GT_PREFIX}" \
  --iou-thr "${IOU_THR}"

echo
echo "✅ YOLO-only Segmentation abgeschlossen."
echo "➡️  NPZ-Masken: ${MASKS_OUT}"
echo "📒 Logs: pipeline/logs/"
