import json
import os

from sklearn import pipeline

def calc_confusion_matrix(path):
    # JSON-Datei laden
    with open(path, 'r') as f:
        data = json.load(f)

    expected_total = data['expected_total']
    true_positives = data['found_total']
    false_negatives = data['false_negatives']
    false_positives = data['false_positives']
    amount_of_images = len(data['per_image'])

    true_negatives = amount_of_images - expected_total - false_positives

    # Precision, Recall, Accuracy und F1-Score berechnen
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / amount_of_images if amount_of_images > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    model_size = data['model_size'] 
    n_train = data['n_train']

    # mean_iou_gt_pos auslesen, falls vorhanden
    mean_iou = data.get('mean_iou_gt_pos', None)

    # Output-Dateiname erstellen (ohne .json)
    output_filename = f"confusion_matrix_{os.path.splitext(os.path.basename(path))[0]}.csv"

    # Speichern der Ergebnisse in einer CSV-Datei
    with open(output_filename, 'w') as f:
        f.write("model_size,n_train,expected_total,true_positives,false_negatives,false_positives,true_negatives,precision,recall,accuracy,f1_score,mean_iou\n")
        f.write(f"{model_size},{n_train},{expected_total},{true_positives},{false_negatives},{false_positives},{true_negatives},{precision},{recall},{accuracy},{f1_score},{mean_iou}\n")
    print(f"Confusion matrix saved to {output_filename}")

if __name__ == "__main__":
    # calc_confusion_matrix("input.json")
    # input1= "inputs/Modelcustom_nTrain1.json"
    # input2= "inputs/Modelcustom_nTrain2.json"
    # input3= "inputs/Modelcustom_nTrain3.json"
    # input4= "inputs/Modelcustom_nTrain4.json"
    # input5= "inputs/Modelcustom_nTrain5.json"
    # input6= "inputs/Modelcustom_nTrain6.json"
    # input7= "inputs/Modelcustom_nTrain7.json"



    input1 = "pipeline\logs\only_sam\Modelsmall_nTrain1_2025-09-09_16-29-37.json"
    input2 = "pipeline\logs\only_sam\Modelsmall_nTrain2_2025-09-09_16-30-31.json"
    input3 = "pipeline\logs\only_sam\Modelsmall_nTrain3_2025-09-09_16-32-19.json"
    input4 = "pipeline\logs\only_sam\Modelsmall_nTrain4_2025-09-09_16-41-10.json"
    input5 = "pipeline\logs\only_sam\Modelsmall_nTrain5_2025-09-09_16-52-11.json"
    input6 = "pipeline\logs\only_sam\Modelsmall_nTrain6_2025-09-09_17-05-34.json"
    input7 = "pipeline\logs\only_sam\Modelsmall_nTrain7_2025-09-09_17-21-16.json"

    calc_confusion_matrix(input1)
    calc_confusion_matrix(input2)
    calc_confusion_matrix(input3)
    calc_confusion_matrix(input4)
    calc_confusion_matrix(input5)
    calc_confusion_matrix(input6)
    calc_confusion_matrix(input7)
    # calc_confusion_matrix(input5)
    # calc_confusion_matrix(input6)
    # calc_confusion_matrix(input7)