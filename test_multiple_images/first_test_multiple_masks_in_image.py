import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor

image_path = os.path.join("combined_train_image.JPG")
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
device = "cuda" if torch.cuda.is_available() else "cpu"

def show_mask(mask, ax, color=None):
    if color is None:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[np.array(labels) == 1]
    neg_points = coords[np.array(labels) == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25, label="Positiv")
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25, label="Negativ")

# Listen für mehrere Masken
all_points = []
all_labels = []
all_masks = []
mask_colors = [
    np.array([1.0, 0.0, 0.0, 0.5]),   # Rot
    np.array([0.0, 1.0, 0.0, 0.5]),   # Grün
    np.array([0.0, 0.0, 1.0, 0.5]),   # Blau
    np.array([1.0, 1.0, 0.0, 0.5]),   # Gelb
    np.array([1.0, 0.0, 1.0, 0.5]),   # Magenta
    np.array([0.0, 1.0, 1.0, 0.5]),   # Cyan
]

clicked_points = []
clicked_labels = []
current_mask = None
current_color_idx = 0

def on_click(event):
    global current_mask
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
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
        current_mask = masks[0]
        redraw()

def on_key(event):
    global clicked_points, clicked_labels, current_mask, current_color_idx
    if event.key == "tab":
        # Neue Maske starten
        if current_mask is not None:
            all_points.append(np.array(clicked_points))
            all_labels.append(np.array(clicked_labels))
            all_masks.append(current_mask)
            current_color_idx = (current_color_idx + 1) % len(mask_colors)
        clicked_points = []
        clicked_labels = []
        current_mask = None
        redraw()
    elif event.key == "enter":
        # Letzte Maske speichern
        if current_mask is not None:
            all_points.append(np.array(clicked_points))
            all_labels.append(np.array(clicked_labels))
            all_masks.append(current_mask)
        plt.close()

def redraw():
    ax.clear()
    ax.imshow(image_np)
    # Vorherige Masken
    for i, mask in enumerate(all_masks):
        show_mask(mask, ax, color=mask_colors[i % len(mask_colors)])
    # Aktuelle Maske
    if current_mask is not None:
        show_mask(current_mask, ax, color=mask_colors[len(all_masks) % len(mask_colors)])
    # Punkte
    for i, (pts, lbls) in enumerate(zip(all_points, all_labels)):
        show_points(pts, lbls, ax)
    if clicked_points:
        show_points(np.array(clicked_points), np.array(clicked_labels), ax)
    ax.set_title(
        f"{len(all_masks) + (1 if current_mask is not None else 0)} Masken. "
        "Tab = neue Maske, Enter = speichern"
    )
    plt.axis('off')
    fig.canvas.draw()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image_np)
ax.set_title("Klicke ins Bild, um Punkte zu setzen\nTab = neue Maske, Enter = speichern")
plt.axis('off')
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

# Nach der Interaktion: Masken speichern
from PIL import Image as PILImage
os.makedirs("erste_anwendung/masks", exist_ok=True)
for i, mask in enumerate(all_masks):
    mask_to_save = (mask > 0).astype(np.uint8) * 255
    pil_mask = PILImage.fromarray(mask_to_save)
    pil_mask.save(f"erste_anwendung/masks/finale_maske_{i+1}.png")
print(f"{len(all_masks)} Masken gespeichert in erste_anwendung/masks/")