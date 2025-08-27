#!/usr/bin/env bash
set -euo pipefail

######################## DAS HIER NUR 1x AUSFÜHREN
# chmod +x pipeline/run_pipeline_configs_zero_shot.sh

#################### HIERMIT SKRIPT AUSFÜHREN
# bash pipeline/run_pipeline_configs_zero_shot.sh

# ===================== USER-CONFIG =====================
SEQ_FOLDER="pipeline/resources/sequence_to_test_2"
MASKS_BASE="pipeline/resources/generated_npz_masks_from_run"
MODELS=( tiny small base-plus large )
MODELS=( tiny )

# ---- NEU: optionale GT-Masken & IoU-Settings ----
GT_MASK_DIR="pipeline/resources/sequence_to_test_2_npz_masks_ground_truth"  # "" lassen oder Ordner weglassen, wenn keine GT vorhanden
GT_PREFIX="train_mask_"     # Prefix der GT-Dateien (z.B. train_mask_12345.npz)
IOU_THR=0.5                 # IoU-Schwelle für 'lokal korrekt'
MIN_PIXELS=1000             # segmentierte Masken kleiner als N Pixel ignorieren
# =======================================================

# Eval-Argumente zusammenstellen
EXTRA_EVAL_ARGS=( --iou-thr "${IOU_THR}" --min-pixels "${MIN_PIXELS}" --gt-prefix "${GT_PREFIX}" )
if [ -n "${GT_MASK_DIR}" ] && [ -d "${GT_MASK_DIR}" ]; then
  EXTRA_EVAL_ARGS+=( --gt-mask-dir "${GT_MASK_DIR}" )
  echo "ℹ️  GT-Masken werden aus '${GT_MASK_DIR}' für IoU/Lokalisierung verwendet (Prefix: '${GT_PREFIX}')."
else
  echo "ℹ️  Keine GT-Masken gefunden – es wird nur TP/TN/FP/FN ohne IoU/Lokalisierung geloggt."
fi
echo

for MODEL in "${MODELS[@]}"; do
  for N in {1..7}; do
    MASKS_FOLDER="${MASKS_BASE}/${MODEL}_n${N}"   # Zielordner für diesen Run
    mkdir -p "${MASKS_FOLDER}"

    echo "=========================================================="
    echo "➡️  Starte: model_size=${MODEL} | n_train=${N}"
    echo "----------------------------------------------------------"

    python3 -m pipeline.cli \
      --model-size   "${MODEL}" \
      --n-train      "${N}" \
      --seq-folder   "${SEQ_FOLDER}" \
      --masks-folder "${MASKS_FOLDER}" \
      "${EXTRA_EVAL_ARGS[@]}"

    echo "✅  Fertig:  model_size=${MODEL} | n_train=${N}"
    echo
  done
done

echo "🎉 Alle Kombinationen abgeschlossen."
