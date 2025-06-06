# 1. Bibliotheken importieren
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 2. Bild laden
# Das Bild wird von der Festplatte geladen und in ein RGB-Bild umgewandelt.
image_path = r"testbilder\CLXQ7779.JPG"
image = Image.open(image_path).convert("RGB")  # Bild öffnen und in RGB konvertieren
image_np = np.array(image)  # In ein numpy-Array umwandeln (für das Modell)

# 3. Device wählen (GPU oder CPU)
# Das Modell läuft schneller auf einer CUDA-fähigen GPU, sonst auf der CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 4. Predictor laden (wie in deinen Beispielen)
# Das kleine (tiny) Modell wird direkt von Hugging Face geladen.
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")

# 5. Mask Generator initialisieren
# Der automatische Maskengenerator nutzt das geladene Modell.
mask_generator = SAM2AutomaticMaskGenerator(predictor.model)

# 6. Masken generieren
# Mit dem automatischen Generator werden alle Masken im Bild erkannt.
with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device=="cuda" else torch.float32):
    masks = mask_generator.generate(image_np)

print(f"{len(masks)} Masken generiert.")

# 7. Hilfsfunktion zum Anzeigen der Masken
def show_anns(anns):
    """
    Zeigt alle generierten Masken als farbige Overlays auf dem Bild an.
    Jede Maske bekommt eine zufällige Farbe mit 50% Transparenz.
    """
    if len(anns) == 0:
        return
    # Nach Fläche sortieren, damit große Masken unten liegen
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # Leeres Bild für die Masken (RGBA)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # Alpha-Kanal auf 0 (voll transparent)
    for ann in sorted_anns:
        m = ann['segmentation']  # Binäre Maske
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # Zufällige Farbe, 50% Transparenz
        img[m] = color_mask  # Maske einfärben
    ax.imshow(img)  # Masken anzeigen

# 8. Visualisierung
plt.figure(figsize=(10, 10))
plt.imshow(image_np)      # Originalbild anzeigen
show_anns(masks)  
        # Masken als Overlay anzeigen
plt.axis('off')
plt.title("Automatisch generierte Masken")
plt.show()

# #save masks to png
# output_path = "output_masks.png"
# plt.imsave(output_path, img)  # Speichert die Masken als PNG
# # Hinweis: Die Masken werden als RGBA-Bild gespeichert, wobei der Alpha-Kanal die Transparenz steuert.  

