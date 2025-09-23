import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_dir = "output/images"
mask_dir = "output/masks"

img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
img_files.sort()

for fname in img_files:
    img_path = os.path.join(img_dir, fname)
    mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Maske als Graustufen

    img_np = np.array(img)
    mask_np = np.array(mask)

    # Erzeuge ein rotes Overlay, wo Maske > 0
    overlay = np.zeros_like(img_np, dtype=np.uint8)
    overlay[..., 0] = 255  # Rotkanal
    alpha = 0.4  # Transparenz

    # Maske als bool-Array
    mask_bool = mask_np > 0

    # Overlay anwenden
    img_overlay = img_np.copy()
    img_overlay[mask_bool] = (alpha * overlay[mask_bool] + (1 - alpha) * img_np[mask_bool]).astype(np.uint8)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_overlay)
    plt.title(fname)
    plt.axis('off')
    plt.show()