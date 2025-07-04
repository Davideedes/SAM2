
# chmod +x pipeline/run_pipeline_configs_finetuned.sh   
# bash pipeline/run_pipeline_configs_finetuned.sh


set -euo pipefail


SEQ_FOLDER="pipeline/resources/sequence_to_test_1"
MASKS_BASE="pipeline/resources/generated_npz_masks_custom"

# CFG_PATH="configs/sam2.1/sam2.1_hiera_t"
# CKPT_PATH="checkpoints/sam2.1_hiera_tiny.pt"
CFG_PATH="sam2_hiera_t"
CKPT_PATH="checkpoints/fine_tuned_sam2_6000.pt"

     # Passe den Pfad ggf. an!

for N in {1..7}; do
  MASKS_FOLDER="${MASKS_BASE}/custom_n${N}"     # Zielordner für diesen Run
  mkdir -p "${MASKS_FOLDER}"                    # Anlegen, falls nötig

  echo "=========================================================="
  echo "➡️  Starte: model_size=custom | n_train=${N}"
  echo "----------------------------------------------------------"

  python3 -m pipeline.cli \
      --model-size   custom \
      --n-train      "${N}" \
      --seq-folder   "${SEQ_FOLDER}" \
      --masks-folder "${MASKS_FOLDER}" \
      --cfg-path     "${CFG_PATH}" \
      --ckpt-path    "${CKPT_PATH}"

  echo "✅  Fertig:  model_size=custom | n_train=${N}"
  echo
done

echo "🎉 Alle Custom-Konfigurationen abgeschlossen."