
import os
import json
import matplotlib.pyplot as plt

# Ordner mit den JSON-Dateien
json_dir = r"pipeline\logs\yolo_and_sam"

# Alle JSON-Dateien im Ordner finden
json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

# Ergebnisse sammeln
results = {}

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    modelsize = data.get("modelsize") or data.get("model_size") or "unknown"
    # Werte auslesen
    tp = data.get("true_positives", 0)
    tn = data.get("true_negatives", 0)
    fp = data.get("false_positives", 0)
    fn = data.get("false_negatives", 0)
    miou = data.get("mean_iou_gt_pos", 0)
    # Präzision, Recall, F1, Accuracy berechnen
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    # Ergebnisse speichern
    if modelsize not in results:
        results[modelsize] = {"miou": [], "f1": [], "precision": [], "recall": [], "accuracy": []}
    results[modelsize]["miou"].append(miou)
    results[modelsize]["f1"].append(f1)
    results[modelsize]["precision"].append(precision)
    results[modelsize]["recall"].append(recall)
    results[modelsize]["accuracy"].append(accuracy)

# Plotten
# Definiere die gewünschte Reihenfolge der Modellgrößen
model_order = ["tiny", "small", "base", "base-plus", "large"]
# Filtere und sortiere die vorhandenen Modelle nach dieser Reihenfolge
model_sizes = [m for m in model_order if m in results] + [m for m in results if m not in model_order]
accuracy_means = [sum(results[m]["accuracy"])/len(results[m]["accuracy"]) if results[m]["accuracy"] else 0 for m in model_sizes]
miou_means = [sum(results[m]["miou"])/len(results[m]["miou"]) if results[m]["miou"] else 0 for m in model_sizes]
f1_means = [sum(results[m]["f1"])/len(results[m]["f1"]) if results[m]["f1"] else 0 for m in model_sizes]
precision_means = [sum(results[m]["precision"])/len(results[m]["precision"]) if results[m]["precision"] else 0 for m in model_sizes]
recall_means = [sum(results[m]["recall"])/len(results[m]["recall"]) if results[m]["recall"] else 0 for m in model_sizes]

plt.figure(figsize=(10,6))
# Farben und Reihenfolge wie im Beispielplot: Accuracy (Blau), Precision (Orange), Recall (Grün), F1-Score (Rot), Mean IoU (Lila)
plt.plot(model_sizes, accuracy_means, marker="o", color="blue", label="Accuracy")
plt.plot(model_sizes, precision_means, marker="o", color="orange", label="Precision")
plt.plot(model_sizes, recall_means, marker="o", color="green", label="Recall")
plt.plot(model_sizes, f1_means, marker="o", color="red", label="F1 score")
plt.plot(model_sizes, miou_means, marker="o", color="purple", label="Mean IoU")
plt.xlabel("Model Size")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("modelsize_comparison.png")
plt.show()