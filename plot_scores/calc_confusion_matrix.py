import json
import os

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

    # Output-Dateiname erstellen (ohne .json)
    output_filename = f"confusion_matrix_{os.path.splitext(os.path.basename(path))[0]}.csv"

    # Speichern der Ergebnisse in einer CSV-Datei
    with open(output_filename, 'w') as f:
        f.write("model_size,n_train,expected_total,true_positives,false_negatives,false_positives,true_negatives,precision,recall,accuracy,f1_score\n")
        f.write(f"{model_size},{n_train},{expected_total},{true_positives},{false_negatives},{false_positives},{true_negatives},{precision},{recall},{accuracy},{f1_score}\n")
    print(f"Confusion matrix saved to {output_filename}")

if __name__ == "__main__":
    # calc_confusion_matrix("input.json")
    input1= "inputs/Modeltiny_nTrain1.json"
    input2= "inputs/Modeltiny_nTrain2.json"
    input3= "inputs/Modeltiny_nTrain3.json"
    input4= "inputs/Modeltiny_nTrain4.json"
    input5= "inputs/Modeltiny_nTrain5.json"
    input6= "inputs/Modeltiny_nTrain6.json"
    input7= "inputs/Modeltiny_nTrain7.json"
    calc_confusion_matrix(input1)
    calc_confusion_matrix(input2)
    calc_confusion_matrix(input3)
    calc_confusion_matrix(input4)
    calc_confusion_matrix(input5)
    calc_confusion_matrix(input6)
    calc_confusion_matrix(input7)