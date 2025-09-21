import matplotlib.pyplot as plt
import re
import os
from datetime import datetime

def parse_training_data(file_path):
    """
    Parse die Trainingsdaten aus der TXT-Datei.
    
    Args:
        file_path (str): Pfad zur TXT-Datei
        
    Returns:
        dict: Dictionary mit den geparsten Daten
    """
    training_data = {}
    validation_data = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            # Pattern für Training-Zeilen
            if 'Valid_IoU' not in line and 'Valid_Seg Loss' not in line:
                # Training data
                match = re.search(r'Step (\d+): Current LR = ([\d.]+), IoU = ([\d.]+), Seg Loss = ([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    lr = float(match.group(2))
                    iou = float(match.group(3))
                    seg_loss = float(match.group(4))
                    
                    training_data[step] = {
                        'lr': lr,
                        'iou': iou,
                        'seg_loss': seg_loss
                    }
            else:
                # Validation data
                match = re.search(r'Step (\d+): Current LR = ([\d.]+), Valid_IoU = ([\d.]+), Valid_Seg Loss = ([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    valid_iou = float(match.group(3))
                    valid_seg_loss = float(match.group(4))
                    
                    validation_data[step] = {
                        'valid_iou': valid_iou,
                        'valid_seg_loss': valid_seg_loss
                    }
    
    # Nur Steps verwenden, die sowohl Training- als auch Validation-Daten haben
    common_steps = sorted(set(training_data.keys()) & set(validation_data.keys()))
    
    data = {
        'steps': common_steps,
        'lr': [training_data[step]['lr'] for step in common_steps],
        'iou': [training_data[step]['iou'] for step in common_steps],
        'seg_loss': [training_data[step]['seg_loss'] for step in common_steps],
        'valid_iou': [validation_data[step]['valid_iou'] for step in common_steps],
        'valid_seg_loss': [validation_data[step]['valid_seg_loss'] for step in common_steps]
    }
    
    return data

def plot_training_metrics(file_path):
    """
    Plotte die Trainingsdaten aus der TXT-Datei.
    
    Args:
        file_path (str): Pfad zur TXT-Datei
    """
    # Daten parsen
    data = parse_training_data(file_path)
    
    # Ordner für die Plots erstellen
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("training_plots", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Learning Rate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['steps'], data['lr'], linestyle='-', color='red', linewidth=2, markersize=4)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title('Learning Rate over Training Steps', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, max(data['steps']) + 1, 1000), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # IoU Plot (Training + Validation)
    plt.figure(figsize=(10, 6))
    plt.plot(data['steps'], data['iou'], linestyle='-', color='blue', linewidth=2, markersize=4, label='Training IoU')
    plt.plot(data['steps'], data['valid_iou'], linestyle='-', color='green', linewidth=2, markersize=4, label='Validation IoU')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('IoU', fontsize=14)
    plt.title('IoU over Training Steps', fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 0), loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, max(data['steps']) + 1, 1000), fontsize=12)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # Segmentation Loss Plot (Training + Validation)
    plt.figure(figsize=(10, 6))
    plt.plot(data['steps'], data['seg_loss'], linestyle='-', color='purple', linewidth=2, markersize=4, label='Training Seg Loss')
    plt.plot(data['steps'], data['valid_seg_loss'],  linestyle='-', color='orange', linewidth=2, markersize=4, label='Validation Seg Loss')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Segmentation Loss', fontsize=14)
    plt.title('Segmentation Loss over Training Steps', fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 0), loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, max(data['steps']) + 1, 1000), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segmentation_loss.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    # Combined Plot - alle Metriken
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 8))

    # Learning Rate
    ax1.plot(data['steps'], data['lr'], linestyle='-', color='red', linewidth=2, markersize=3)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title('Learning Rate', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, max(data['steps']) + 1, 1000))

    # IoU
    ax2.plot(data['steps'], data['iou'], linestyle='-', color='blue', linewidth=2, markersize=3, label='Training')
    ax2.plot(data['steps'], data['valid_iou'], linestyle='-', color='green', linewidth=2, markersize=3, label='Validation')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('IoU', fontsize=12)
    ax2.set_title('IoU', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, max(data['steps']) + 1, 1000))
    ax2.set_ylim(0, 1.05)

    # Segmentation Loss
    ax3.plot(data['steps'], data['seg_loss'], linestyle='-', color='purple', linewidth=2, markersize=3, label='Training')
    ax3.plot(data['steps'], data['valid_seg_loss'], linestyle='-', color='orange', linewidth=2, markersize=3, label='Validation')
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Segmentation Loss', fontsize=12)
    ax3.set_title('Segmentation Loss', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, max(data['steps']) + 1, 1000))

    # # Gap zwischen Training und Validation IoU (zeigt Overfitting)
    # iou_gap = [abs(train - val) for train, val in zip(data['iou'], data['valid_iou'])]
    # ax4.plot(data['steps'], iou_gap, marker='o', linestyle='-', color='red', linewidth=2, markersize=3)
    # ax4.set_xlabel('Training Steps', fontsize=12)
    # ax4.set_ylabel('IoU Gap (|Train - Val|)', fontsize=12)
    # ax4.set_title('Training-Validation IoU Gap', fontsize=14)
    # ax4.grid(True, alpha=0.3)
    # ax4.set_xticks(range(0, max(data['steps']) + 1, 1000))
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Training plots wurden im Ordner '{output_dir}' gespeichert.")
    print(f"Anzahl der verarbeiteten Trainingssteps: {len(data['steps'])}")

if __name__ == "__main__":
    # Datei plotten
    plot_training_metrics("training_values.txt")