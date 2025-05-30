import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Bild laden
image_path = r"testbilder\CLXQ7779.JPG"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Device wählen
device = "cuda" if torch.cuda.is_available() else "cpu"

# Predictor laden (wie in deinen Beispielen)
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")

# Mask Generator initialisieren
mask_generator = SAM2AutomaticMaskGenerator(predictor.model)

# Masken generieren
with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device=="cuda" else torch.float32):
    masks = mask_generator.generate(image_np)

print(f"{len(masks)} Masken generiert.")

# Hilfsfunktion zum Anzeigen der Masken
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)

# Visualisierung
plt.figure(figsize=(10, 10))
plt.imshow(image_np)
show_anns(masks)
plt.axis('off')
plt.title("Automatisch generierte Masken")
plt.show()