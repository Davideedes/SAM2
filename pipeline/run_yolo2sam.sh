#!/usr/bin/env bash
set -euo pipefail

# ============= USER-CONFIG ==========================
SEQ_FOLDER="pipeline/resources/sequence_to_test_1"   # muss zu YOLOs SOURCE passen!
SAM_MODELS=( tiny small base-plus large )
# SAM_MODELS=( tiny )
MASKS_BASE="pipeline/resources/generated_npz_masks_yolo2sam"
MAX_SIDE=0
USE_CUSTOM_SAM=false
SAM_CFG_PATH="configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT_PATH="checkpoints/sam2.1_hiera_tiny.pt"
# ====================================================

echo "=========================================================="
echo "➡️  YOLO: nutze dein Script yolov12/run_yolo.py"
echo "----------------------------------------------------------"

# YOLO ausführen und die 'LABELS_DIR:'-Zeile abfangen
YOLO_LOG=$(python3 yolov12/run_yolo.py)
echo "$YOLO_LOG"

LABELS_DIR=$(echo "$YOLO_LOG" | awk -F': ' '/^LABELS_DIR:/ {print $2; exit}')
if [ -z "${LABELS_DIR:-}" ] || [ ! -d "$LABELS_DIR" ]; then
  echo "❌ Konnte LABELS_DIR nicht finden."
  echo "   Debug: LABELS_DIR='$LABELS_DIR'"
  echo "   Kandidaten unter runs/:"
  find runs -maxdepth 3 -type d -name labels 2>/dev/null | sort || true
  exit 1
fi
echo "✅ Labels gefunden: $LABELS_DIR"

echo
echo "=========================================================="
echo "➡️  SAM2: Segmentiere mit Box-Prompts aus: ${LABELS_DIR}"
echo "----------------------------------------------------------"
for MODEL in "${SAM_MODELS[@]}"; do
  MASKS_FOLDER="${MASKS_BASE}/${MODEL}"
  mkdir -p "${MASKS_FOLDER}"
  echo "---- SAM2 Run: model_size=${MODEL} ----"
  if [ "${USE_CUSTOM_SAM}" = true ]; then
    python -m pipeline.cli_yolo2sam \
      --model-size custom \
      --seq-folder   "${SEQ_FOLDER}" \
      --labels-folder "${LABELS_DIR}" \
      --masks-folder "${MASKS_FOLDER}" \
      --cfg-path     "${SAM_CFG_PATH}" \
      --ckpt-path    "${SAM_CKPT_PATH}" \
      --max-side     "${MAX_SIDE}"
  else
    python -m pipeline.cli_yolo2sam \
      --model-size   "${MODEL}" \
      --seq-folder   "${SEQ_FOLDER}" \
      --labels-folder "${LABELS_DIR}" \
      --masks-folder "${MASKS_FOLDER}" \
      --max-side     "${MAX_SIDE}"
  fi
  echo "✅ Fertig: SAM2 ${MODEL} → ${MASKS_FOLDER}"
done

echo
echo "🎉 YOLO ➜ SAM2 abgeschlossen. Logs: pipeline/logs/"
