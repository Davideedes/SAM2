#!/usr/bin/env bash
set -euo pipefail

# ============= USER-CONFIG ==========================
# (Wird überschrieben, falls YOLO etwas anderes als Quelle loggt)
SEQ_FOLDER="pipeline/resources/sequence_to_test_2"

# Alle gewünschten SAM-Modelle (werden nacheinander ausgeführt)
MODELS=( tiny small base-plus large )

MASKS_BASE="pipeline/resources/temp"
MAX_SIDE=0

USE_CUSTOM_SAM=false
SAM_CFG_PATH="configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT_PATH="checkpoints/sam2.1_hiera_tiny.pt"

# ---- GT-Masken & IoU-Settings ----
GT_MASK_DIR="pipeline/resources/sequence_to_test_2_npz_masks_ground_truth"
GT_PREFIX="train_mask_"
IOU_THR=0.7
MIN_PIXELS=1000
# ====================================================

echo "=========================================================="
echo "➡️  YOLO: nutze dein Script yolov12/run_yolo.py"
echo "----------------------------------------------------------"

# YOLO ausführen und Log abfangen (nur EINMAL)
YOLO_LOG=$(python3 yolov12/run_yolo.py)
echo "$YOLO_LOG"

# LABELS_DIR herausziehen
LABELS_DIR=$(echo "$YOLO_LOG" | awk -F': ' '/^LABELS_DIR:/ {print $2; exit}')
if [ -z "${LABELS_DIR:-}" ] || [ ! -d "$LABELS_DIR" ]; then
  echo "❌ Konnte LABELS_DIR nicht finden."
  echo "   Debug: LABELS_DIR='$LABELS_DIR'"
  echo "   Kandidaten unter runs/:"
  find runs -maxdepth 3 -type d -name labels 2>/dev/null | sort || true
  exit 1
fi
echo "✅ Labels gefunden: $LABELS_DIR"

# SOURCE/SEQ-Folder robust aus dem YOLO-Log holen
# 1) Bevorzugt 'SOURCE_DIR:' (falls dein run_yolo.py das loggt)
SEQ_FROM_LOG=$(echo "$YOLO_LOG" | awk -F': ' '/^SOURCE_DIR:/ {print $2; exit}')
# 2) Fallback: Zeile, die mit 'source' beginnt und dann den Pfad enthält (Format: 'source  : <pfad> True')
if [ -z "${SEQ_FROM_LOG:-}" ]; then
  SEQ_FROM_LOG=$(echo "$YOLO_LOG" | awk '/^source/ {print $3; exit}')
fi
# Wenn gefunden und Verzeichnis existiert → SEQ_FOLDER überschreiben
if [ -n "${SEQ_FROM_LOG:-}" ] && [ -d "$SEQ_FROM_LOG" ]; then
  SEQ_FOLDER="$SEQ_FROM_LOG"
fi

if [ ! -d "$SEQ_FOLDER" ]; then
  echo "❌ SEQ_FOLDER existiert nicht: $SEQ_FOLDER"
  exit 1
fi
echo "📁 SEQ_FOLDER (Bilder für SAM2): $SEQ_FOLDER"

# Eval-Argumente für cli_yolo2sam zusammenstellen
EXTRA_EVAL_ARGS=( --iou-thr "${IOU_THR}" --min-pixels "${MIN_PIXELS}" --gt-prefix "${GT_PREFIX}" )
if [ -n "${GT_MASK_DIR}" ] && [ -d "${GT_MASK_DIR}" ]; then
  EXTRA_EVAL_ARGS+=( --gt-mask-dir "${GT_MASK_DIR}" )
  echo "ℹ️  GT-Masken werden für IoU/Lokalisierung genutzt aus: ${GT_MASK_DIR} (Prefix: ${GT_PREFIX})"
else
  echo "ℹ️  Keine GT-Masken (oder Ordner nicht vorhanden) – es wird nur Bild-TP/TN/FP/FN geloggt."
fi

echo
echo "=========================================================="
echo "➡️  SAM2: Segmentiere mit Box-Prompts aus: ${LABELS_DIR}"
echo "----------------------------------------------------------"

# WICHTIG: Ein Durchlauf über ALLE Modelle
for MODEL in "${MODELS[@]}"; do
  MASKS_FOLDER="${MASKS_BASE}/${MODEL}"
  mkdir -p "${MASKS_FOLDER}"
  echo "---- SAM2 Run: model_size=${MODEL} ----"
  if [ "${USE_CUSTOM_SAM}" = true ]; then
    python -m pipeline.cli_yolo2sam \
      --model-size custom \
      --seq-folder    "${SEQ_FOLDER}" \
      --labels-folder "${LABELS_DIR}" \
      --masks-folder  "${MASKS_FOLDER}" \
      --cfg-path      "${SAM_CFG_PATH}" \
      --ckpt-path     "${SAM_CKPT_PATH}" \
      --max-side      "${MAX_SIDE}" \
      "${EXTRA_EVAL_ARGS[@]}"
  else
    python -m pipeline.cli_yolo2sam \
      --model-size    "${MODEL}" \
      --seq-folder    "${SEQ_FOLDER}" \
      --labels-folder "${LABELS_DIR}" \
      --masks-folder  "${MASKS_FOLDER}" \
      --max-side      "${MAX_SIDE}" \
      "${EXTRA_EVAL_ARGS[@]}"
  fi
  echo "✅ Fertig: SAM2 ${MODEL} → ${MASKS_FOLDER}"
done

echo
echo "🎉 YOLO ➜ SAM2 abgeschlossen. Logs: pipeline/logs/"
