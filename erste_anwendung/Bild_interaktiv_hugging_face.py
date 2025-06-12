# 1. Bibliotheken importieren
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 2. Bild laden
#image_path = os.path.join("testbilder", "testschlagloch.jpg")
image_path = os.path.join("testbilder", "CLXQ7779.JPG")
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# 3. SAM2 Predictor laden (tiny-Modell)
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")

# 4. Device wählen
device = "cuda" if torch.cuda.is_available() else "cpu"

# 5. Hilfsfunktionen für Visualisierung
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1.0, 0.0, 0.0, 0.6])  # Knallrot mit 60% Transparenz
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    # Positive Punkte (Label 1) grün, negative (Label 0) lila
    pos_points = coords[np.array(labels) == 1]
    neg_points = coords[np.array(labels) == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25, label="Positiv")
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25, label="Negativ")  # Lila

# 6. Interaktive Callback-Funktionen
clicked_points = []
clicked_labels = []
final_mask = None  # Hier speichern wir die finale Maske

def on_click(event):
    global final_mask
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        # Linksklick = Vordergrund (grün), Rechtsklick = Hintergrund (lila)
        if event.button == 1:
            clicked_points.append([x, y])
            clicked_labels.append(1)
        elif event.button == 3:
            clicked_points.append([x, y])
            clicked_labels.append(0)
        else:
            return
        input_point = np.array(clicked_points)
        input_label = np.array(clicked_labels)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device=="cuda" else torch.float32):
            predictor.set_image(image_np)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
        final_mask = masks[0]  # Beste Maske merken
        ax.clear()
        ax.imshow(image_np)
        show_mask(masks[0], ax)
        show_points(input_point, input_label, ax)
        ax.set_title(f"{len(clicked_points)} Punkt(e) gesetzt (grün=+, lila=-)\nDrücke Enter zum Speichern")
        plt.axis('off')
        fig.canvas.draw()

def on_key(event):
    if event.key == "enter":
        plt.close()  # Fenster schließen

# 7. Bild anzeigen und auf Klicks/Enter warten
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image_np)
ax.set_title("Klicke ins Bild, um Punkte zu setzen\nDrücke Enter zum Speichern")
plt.axis('off')
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

# Nach der Interaktion: finale Maske speichern
if final_mask is not None:
    from PIL import Image as PILImage
    mask_to_save = (final_mask > 0).astype(np.uint8) * 255
    pil_mask = PILImage.fromarray(mask_to_save)
    pil_mask.save(r"erste_anwendung/masks/finale_maske.png")
    print("Finale Maske gespeichert als finale_maske.png")
