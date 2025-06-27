#!/bin/bash

# Setze heutiges Datum für das Logfile
LOGFILE="seq/runlog_try_different_configs$(date +%Y-%m-%d).txt"

# Optional: setze das Verzeichnis der Sequenzbilder hier
SEQ_FOLDER="seq/meister_bertram_mit_eindeutigen_potholes"

echo "Starte Experimente am $(date)" | tee -a "$LOGFILE"

# Modelgrößen in Reihenfolge
for model_size in tiny small large; do
    for n_train in {1..6}; do
        echo "----------------------------------------" | tee -a "$LOGFILE"
        echo "⏱️  $(date): Modell: $model_size | Trainingsbilder: $n_train" | tee -a "$LOGFILE"
        
        # Der eigentliche Aufruf
        python3 erste_anwendung/cross_image_transfer.py \
            --model_size "$model_size" \
            --n_train "$n_train" \
            --seq_folder "$SEQ_FOLDER" \
            2>&1 | tee -a "$LOGFILE"
        
        echo "" | tee -a "$LOGFILE"
    done
done

echo "✅ Alle Läufe abgeschlossen: $(date)" | tee -a "$LOGFILE"
