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
    # model_size_order = ["tiny", "small", "base-plus", "large"]
    # combined_df["model_size"] = pd.Categorical(combined_df["model_size"], categories=model_size_order, ordered=True)

    model_size_order = ["tiny", "small", "base-plus", "large", "custom"]
    combined_df["model_size"] = pd.Categorical(combined_df["model_size"], categories=model_size_order, ordered=True)
 
    # Metriken, die geplottet werden sollen (inkl. mean_iou)
    metrics = ["true_positives", "false_positives", "true_negatives", "false_negatives", 
               "precision", "recall", "accuracy", "f1_score", "mean_iou"]

    # Funktion zum Plotten für beide x-Achsen
    def plot_for_x_axis(x_axis, output_dir):
        for metric in metrics:
            plt.figure(figsize=(10, 5))  # Kleinere Abbildung
            # Daten sortieren nach der x-Achse für eine saubere Kurve
            combined_df_sorted = combined_df.sort_values(by=x_axis)
            plt.plot(combined_df_sorted[x_axis], combined_df_sorted[metric], marker='o', linestyle='-', color='blue')

            # Beschriftung der x-Achse anpassen
            x_label = "Number of Input Images" if x_axis == "n_train" else "Model Size"
            plt.xlabel(x_label, fontsize=14)  # Größere Schriftgröße
            plt.ylabel(metric.replace("_", " ").capitalize(), fontsize=14)  # Größere Schriftgröße
            # Y-Achse für Performance-Metriken auf 0 bis 1.05 begrenzen
            if metric in ["precision", "recall", "accuracy", "f1_score"]:
                plt.ylim(0, 1.05)
            # Schrittweite für n_train auf der x-Achse
            if x_axis == "n_train":
                plt.xticks(combined_df_sorted[x_axis], fontsize=12)
            plt.grid(True)

            # Dateiname erstellen
            filename = f"{x_axis}_vs_{metric}.svg"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, format="svg")
            plt.close()

        # Kombinierter Plot für tp, tn, fp, fn
        plt.figure(figsize=(10, 5))  # Kleinere Abbildung
        combined_df_sorted = combined_df.sort_values(by=x_axis)
        for metric in ["true_positives", "true_negatives", "false_positives", "false_negatives"]:
            plt.plot(combined_df_sorted[x_axis], combined_df_sorted[metric], marker='o', linestyle='-', label=metric.replace("_", " ").capitalize())
        x_label = "Number of Input Images" if x_axis == "n_train" else "Model Size"
        plt.xlabel(x_label, fontsize=14)  # Größere Schriftgröße
        plt.ylabel("Values", fontsize=14)  # Größere Schriftgröße
        if x_axis == "n_train":
            plt.xticks(combined_df_sorted[x_axis], fontsize=12)
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 0), loc='lower left')  # Legende außerhalb unten rechts
        plt.grid(True)
        filename = f"{x_axis}_vs_confusion_matrix_values.svg"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()  # Layout anpassen für bessere Darstellung
        plt.savefig(filepath, format="svg", bbox_inches='tight')
        plt.close()

        # Kombinierter Plot für Acc, Prec, Recall, F1, mean_iou
        plt.figure(figsize=(10, 5))  # Kleinere Abbildung
        perf_metrics = ["accuracy", "precision", "recall", "f1_score", "mean_iou"]
        perf_colors = {"accuracy": "blue", "precision": "orange", "recall": "green", "f1_score": "red", "mean_iou": "purple"}
        for metric in perf_metrics:
            if metric in combined_df_sorted.columns:
                plt.plot(combined_df_sorted[x_axis], combined_df_sorted[metric], marker='o', linestyle='-', color=perf_colors[metric], label=metric.replace("_", " ").capitalize() if metric != "mean_iou" else "Mean IoU")
        x_label = "Number of Input Images" if x_axis == "n_train" else "Model Size"
        plt.xlabel(x_label, fontsize=14)  # Größere Schriftgröße
        plt.ylabel("Values", fontsize=14)  # Größere Schriftgröße
        plt.ylim(0, 1.05)  # Y-Achse immer von 0 bis 1.05
        if x_axis == "n_train":
            plt.xticks(combined_df_sorted[x_axis], fontsize=12)
        plt.legend(fontsize=14, bbox_to_anchor=(1.05, 0), loc='lower left')  # Legende außerhalb unten rechts
        plt.grid(True)
        filename = f"{x_axis}_vs_performance_metrics.svg"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()  # Layout anpassen für bessere Darstellung
        plt.savefig(filepath, format="svg", bbox_inches='tight')
        plt.close()

    # Plots für beide x-Achsen erstellen
    plot_for_x_axis("n_train", n_train_dir)
    plot_for_x_axis("model_size", model_size_dir)

    print(f"Plots wurden im Ordner '{base_output_dir}' gespeichert.")

# Beispielaufruf
# size = "tiny"
# input1= "Modelcustom_nTrain1"
# input2= "Modelcustom_nTrain2"
# input3= "Modelcustom_nTrain3"
# input4= "Modelcustom_nTrain4"
# input5= "Modelcustom_nTrain5"
# input6= "Modelcustom_nTrain6"
# input7= "Modelcustom_nTrain7"

# input1 = "Modeltiny_nTrain6"
# input2 = "Modelsmall_nTrain6"
# input3 = "Modelbase-plus_nTrain6"
# input4 = "Modellarge_nTrain6"


# Diese sollen geplottet werden:
    # input1 = "pipeline\logs\only_sam\Modelsmall_nTrain1_2025-09-09_16-29-37.json"
    # input2 = "pipeline\logs\only_sam\Modelsmall_nTrain2_2025-09-09_16-30-31.json"
    # input3 = "pipeline\logs\only_sam\Modelsmall_nTrain3_2025-09-09_16-32-19.json"
    # input4 = "pipeline\logs\only_sam\Modelsmall_nTrain4_2025-09-09_16-41-10.json"
    # input5 = "pipeline\logs\only_sam\Modelsmall_nTrain5_2025-09-09_16-52-11.json"
    # input6 = "pipeline\logs\only_sam\Modelsmall_nTrain6_2025-09-09_17-05-34.json"
    # input7 = "pipeline\logs\only_sam\Modelsmall_nTrain7_2025-09-09_17-21-16.json"

    #     input1 = "pipeline\logs\only_sam_seq1_new_2209\Modelbase-plus_nTrain1_2025-09-22_19-51-34.json"
    # input2 = "pipeline\logs\only_sam_seq1_new_2209\Modelbase-plus_nTrain2_2025-09-22_19-53-18.json"
    # input3 = "pipeline\logs\only_sam_seq1_new_2209\Modelbase-plus_nTrain3_2025-09-22_19-56-45.json"
    # input4 = "pipeline\logs\only_sam_seq1_new_2209\Modelbase-plus_nTrain4_2025-09-22_20-01-32.json"
    # input5 = "pipeline\logs\only_sam_seq1_new_2209\Modelbase-plus_nTrain5_2025-09-22_20-07-29.json"
    # input6 = "pipeline\logs\only_sam_seq1_new_2209\Modelbase-plus_nTrain6_2025-09-22_20-14-42.json"
    # input7 = "pipeline\logs\only_sam_seq1_new_2209\Modelbase-plus_nTrain7_2025-09-22_20-23-12.json"

    


# input1 = "Modelbase-plus_nTrain1_2025-09-22_19-51-34"
# input2 = "Modelbase-plus_nTrain2_2025-09-22_19-53-18"
# input3 = "Modelbase-plus_nTrain3_2025-09-22_19-56-45"
# input4 = "Modelbase-plus_nTrain4_2025-09-22_20-01-32"
# input5 = "Modelbase-plus_nTrain5_2025-09-22_20-07-29"
# input6 = "Modelbase-plus_nTrain6_2025-09-22_20-14-42"
# input7 = "Modelbase-plus_nTrain7_2025-09-22_20-23-12"
# input5 = "Modelcustom_nTrain5"

    # input1 = "pipeline\logs\jannik_new_standard_settings\Modeltiny_nTrain6_2025-09-23_11-06-02.json"
    # input2 = "pipeline\logs\jannik_new_standard_settings\Modelsmall_nTrain6_2025-09-23_11-16-39.json"
    # input3 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain6_2025-09-23_11-29-31.json"
    # input4 = "pipeline\logs\jannik_new_standard_settings\Modellarge_nTrain6_2025-09-23_11-49-56.json"
# Jannik Model size
# input1 = "Modeltiny_nTrain6_2025-09-23_11-06-02"
# input2 = "Modelsmall_nTrain6_2025-09-23_11-16-39"
# input3 = "Modelbase-plus_nTrain6_2025-09-23_11-29-31"
# input4 = "Modellarge_nTrain6_2025-09-23_11-49-56"

# Jannik nTrain base plus 
    # input1 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain1_2025-09-23_11-21-11.json"
    # input2 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain2_2025-09-23_11-22-02.json"
    # input3 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain3_2025-09-23_11-23-21.json"
    # input4 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain4_2025-09-23_11-25-02.json"
    # input5 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain5_2025-09-23_11-27-04.json"
    # input6 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain6_2025-09-23_11-29-31.json"
    # input7 = "pipeline\logs\jannik_new_standard_settings\Modelbase-plus_nTrain7_2025-09-23_11-32-17.json"

input1= "Modelbase-plus_nTrain1_2025-09-23_11-21-11"
input2= "Modelbase-plus_nTrain2_2025-09-23_11-22-02"
input3= "Modelbase-plus_nTrain3_2025-09-23_11-23-21"
input4= "Modelbase-plus_nTrain4_2025-09-23_11-25-02"
input5= "Modelbase-plus_nTrain5_2025-09-23_11-27-04"
input6= "Modelbase-plus_nTrain6_2025-09-23_11-29-31"
input7= "Modelbase-plus_nTrain7_2025-09-23_11-32-17"

# paths = [input1, input2, input3, input4]
paths = [input1, input2, input3, input4, input5, input6, input7]
# paths = [input1, input2, input3, input4]

plot_metrics(paths)
