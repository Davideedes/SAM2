
# chmod +x pipeline/run_pipeline_configs_zero_shot.sh   
# bash pipeline/run_pipeline_configs_zero_shot.sh


#!/usr/bin/env bash
set -euo pipefail

SEQ_FOLDER="pipeline/resources/sequence_to_test_1"
MASKS_BASE="pipeline/resources/generated_npz_masks_from_run"
MODELS=( tiny small base large )

for MODEL in "${MODELS[@]}"; do
  for N in {1..7}; do
    MASKS_FOLDER="${MASKS_BASE}/${MODEL}_n${N}"   # Zielordner für diesen Run
    mkdir -p "${MASKS_FOLDER}"                    # anlegen falls nicht vorhanden

    echo "=========================================================="
    echo "➡️  Starte: model_size=${MODEL} | n_train=${N}"
    echo "----------------------------------------------------------"

    python3 -m pipeline.cli \
        --model-size   "${MODEL}" \
        --n-train      "${N}" \
        --seq-folder   "${SEQ_FOLDER}" \
        --masks-folder "${MASKS_FOLDER}"

    echo "✅  Fertig:  model_size=${MODEL} | n_train=${N}"
    echo
  done
done

echo "🎉 Alle Kombinationen abgeschlossen."