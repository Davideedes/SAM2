import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def plot_metrics(paths):
    """
    Plotte die Metriken aus den CSV-Dateien und speichere die Plots als SVG.

    Args:
        paths (list): Liste von Pfaden zu den CSV-Dateien.
    """
    # Ordner für die Plots erstellen
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = os.path.join("plots", timestamp)
    os.makedirs(base_output_dir, exist_ok=True)

    # Unterordner für n_train und model_size erstellen
    n_train_dir = os.path.join(base_output_dir, "n_train")
    model_size_dir = os.path.join(base_output_dir, "model_size")
    os.makedirs(n_train_dir, exist_ok=True)
    os.makedirs(model_size_dir, exist_ok=True)

    # Daten aus den CSV-Dateien laden
    dfs = []
    for path in paths:
        df = pd.read_csv(f"confusion_matrix_{path}.csv")
        dfs.append(df)

    # Alle Daten zusammenführen
    combined_df = pd.concat(dfs, ignore_index=True)

    # Reihenfolge für model_size definieren
    model_size_order = ["tiny", "small", "base_plus", "large"]
    combined_df["model_size"] = pd.Categorical(combined_df["model_size"], categories=model_size_order, ordered=True)

    # Metriken, die geplottet werden sollen
    metrics = ["true_positives", "false_positives", "true_negatives", "false_negatives", 
               "precision", "recall", "accuracy", "f1_score"]

    # Funktion zum Plotten für beide x-Achsen
    def plot_for_x_axis(x_axis, output_dir):
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            # Daten sortieren nach der x-Achse für eine saubere Kurve
            combined_df_sorted = combined_df.sort_values(by=x_axis)
            plt.plot(combined_df_sorted[x_axis], combined_df_sorted[metric], marker='o', linestyle='-', color='blue')

            plt.xlabel(x_axis.replace("_", " ").capitalize())
            plt.ylabel(metric.replace("_", " ").capitalize())
            # Y-Achse für Performance-Metriken auf 0 bis 1 begrenzen
            if metric in ["precision", "recall", "accuracy", "f1_score"]:
                plt.ylim(0, 1)
            # plt.title(f"{metric.replace('_', ' ').capitalize()} vs {x_axis.replace('_', ' ').capitalize()}")
            plt.grid(True)

            # Dateiname erstellen
            filename = f"{x_axis}_vs_{metric}.svg"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, format="svg")
            plt.close()

        # Kombinierter Plot für tp, tn, fp, fn
        plt.figure(figsize=(10, 6))
        combined_df_sorted = combined_df.sort_values(by=x_axis)
        for metric in ["true_positives", "true_negatives", "false_positives", "false_negatives"]:
            plt.plot(combined_df_sorted[x_axis], combined_df_sorted[metric], marker='o', linestyle='-', label=metric.replace("_", " ").capitalize())
        plt.xlabel(x_axis.replace("_", " ").capitalize())
        plt.ylabel("Values")
        # plt.title(f"Confusion Matrix Values vs {x_axis.replace('_', ' ').capitalize()}")
        plt.legend()
        plt.grid(True)
        filename = f"{x_axis}_vs_confusion_matrix_values.svg"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format="svg")
        plt.close()

        # Kombinierter Plot für Acc, Prec, Recall, F1
        plt.figure(figsize=(10, 6))
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            plt.plot(combined_df_sorted[x_axis], combined_df_sorted[metric], marker='o', linestyle='-', label=metric.replace("_", " ").capitalize())
        plt.xlabel(x_axis.replace("_", " ").capitalize())
        plt.ylabel("Metrics")
        plt.ylim(0, 1)  # Y-Achse immer von 0 bis 1
        # plt.title(f"Performance Metrics vs {x_axis.replace('_', ' ').capitalize()}")
        plt.legend()
        plt.grid(True)
        filename = f"{x_axis}_vs_performance_metrics.svg"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format="svg")
        plt.close()

    # Plots für beide x-Achsen erstellen
    plot_for_x_axis("n_train", n_train_dir)
    plot_for_x_axis("model_size", model_size_dir)

    print(f"Plots wurden im Ordner '{base_output_dir}' gespeichert.")

# Beispielaufruf
paths = ["input.json", "input2.json"]
plot_metrics(paths)