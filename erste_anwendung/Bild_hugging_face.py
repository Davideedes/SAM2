# 1. Bibliotheken importieren
# Wir benötigen PyTorch für das Modell, PIL für das Laden von Bildern, numpy für die Array-Verarbeitung,
# und matplotlib für die Visualisierung.
import torch
from PIL import Image
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 2. Bild laden
# Das Bild wird von der Festplatte geladen und in ein RGB-Bild umgewandelt.
image_path = r"testbilder\CLXQ7779.JPG"  # Pfad zum Eingabebild
image = Image.open(image_path).convert("RGB")  # Bild öffnen und in RGB konvertieren
image_np = np.array(image)  # In ein numpy-Array umwandeln (für das Modell)

# 3. SAM2 Predictor laden
# Wir nutzen das kleine (tiny) Modell direkt von Hugging Face.
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")

# 4. Device wählen (GPU oder CPU)
# Das Modell läuft schneller auf einer CUDA-fähigen GPU, sonst auf der CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 5. Prompt für die Segmentierung erzeugen
# Wir wählen einen zufälligen Punkt im Bild als Eingabe für das Modell.
h, w = image_np.shape[:2]  # Höhe und Breite des Bildes bestimmen
input_point = np.array([[np.random.randint(0, w), np.random.randint(0, h)]])  # Zufälliger Punkt (x, y)
input_label = np.array([1])  # Label 1 bedeutet: "Das ist ein Vordergrundpunkt"

# 6. Bild an das Modell übergeben und Maske vorhersagen lassen
# Wir schalten in den Inferenzmodus (kein Training) und nutzen automatische Typanpassung für Effizienz.
with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device=="cuda" else torch.float32):
    predictor.set_image(image_np)  # Bild an den Predictor übergeben (berechnet Embedding)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,      # Die Koordinaten des Prompts
        point_labels=input_label,      # Die Labels des Prompts
        multimask_output=True,         # Mehrere Masken als Ergebnis (für mehr Auswahl)
    )

# 7. Ergebnisse ausgeben
print("masks.shape:", masks.shape)  # Form der Masken (z.B. 3 Masken, Höhe, Breite)
print("scores:", scores)            # Qualitäts-Scores für jede Maske

# 8. Hilfsfunktionen für die Visualisierung
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    """
    Zeigt eine Maske auf einer matplotlib-Achse an.
    Standardfarbe ist knallrot mit 60% Transparenz.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # Zufällige Farbe
    else:
        color = np.array([1.0, 0.0, 0.0, 0.6])  # Knallrot mit 60% Transparenz (RGBA)
    h, w = mask.shape[-2:]  # Höhe und Breite der Maske
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # Maske einfärben
    ax.imshow(mask_image)  # Maske anzeigen

def show_points(coords, labels, ax, marker_size=375):
    """
    Zeigt die Prompt-Punkte als grüne Sterne auf dem Bild an.
    """
    ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size,
               edgecolor='white', linewidth=1.25)

# 9. Visualisierung der Ergebnisse
plt.figure(figsize=(10, 10))           # Große Abbildung
plt.imshow(image_np)                    # Originalbild anzeigen
show_mask(masks[0], plt.gca())          # Erste Maske anzeigen (knallrot)
show_points(input_point, input_label, plt.gca())  # Prompt-Punkt grün anzeigen
plt.axis('off')                         # Achsen ausblenden
plt.show()                              # Bild anzeigen