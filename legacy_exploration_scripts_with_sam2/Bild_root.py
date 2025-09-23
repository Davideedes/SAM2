# 1. Bibliotheken importieren
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 2. Bild laden
image_path = r"testbilder\CLXQ7779.JPG"  # Pfad zum Eingabebild
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# 3. Device wählen (GPU oder CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 4. SAM2 tiny-Modell und Predictor vom lokalen Checkpoint laden
sam2_checkpoint = "checkpoints\sam2.1_hiera_tiny.pt"  # Passe den Pfad ggf. an!
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"      # Passe den Pfad ggf. an!
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# 5. Prompt für die Segmentierung erzeugen (zufälliger Punkt)
h, w = image_np.shape[:2]
input_point = np.array([[np.random.randint(0, w), np.random.randint(0, h)]])
input_label = np.array([1])

# 6. Bild an das Modell übergeben und Maske vorhersagen lassen
with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device=="cuda" else torch.float32):
    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

print("masks.shape:", masks.shape)
print("scores:", scores)

# 7. Hilfsfunktionen für die Visualisierung
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1.0, 0.0, 0.0, 0.6])  # Knallrot mit 60% Transparenz
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)

# 8. Visualisierung der Ergebnisse
plt.figure(figsize=(10, 10))
plt.imshow(image_np)
show_mask(masks[0], plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()