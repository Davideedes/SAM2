import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_video_predictor import SAM2VideoPredictor
import torch
import tempfile
import shutil

# Hilfsfunktion: Bild auf 1024x1024 skalieren und mittig croppen
def resize_and_crop(img, size=1024):
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    upper = (new_h - size) // 2
    img = img.crop((left, upper, left + size, upper + size))
    return img

# Eingabe- und Ausgabeordner
input_dir = "train_masks"  # <-- hier Pfad zu deinem Ordner eintragen!
output_dir = "output"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

# SAM2 Video Predictor laden (tiny-Modell)
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Alle Bilddateien im Ordner finden
img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
img_files.sort()  # Optional: alphabetisch sortieren

for fname in img_files:
    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Bild temporär als "Video" mit nur einem Frame speichern
    tmpdir = tempfile.mkdtemp()
    tmp_img_path = os.path.join(tmpdir, "00000.jpg")
    Image.fromarray(img_np).save(tmp_img_path)
    inference_state = predictor.init_state(video_path=tmpdir)

    clicked_points = []
    clicked_labels = []
    final_mask = None

    def show_mask(mask, ax):
        color = np.array([1.0, 0.0, 0.0, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[np.array(labels) == 1]
        neg_points = coords[np.array(labels) == 0]
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                       edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*', s=marker_size,
                       edgecolor='white', linewidth=1.25)

    def on_click(event):
        global final_mask
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
            input_points = np.array(clicked_points, dtype=np.float32)
            input_labels = np.array(clicked_labels, dtype=np.int32)
            # VideoPredictor: add_new_points_or_box für Frame 0
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=input_points,
                labels=input_labels,
            )
            final_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            ax.clear()
            ax.imshow(img_np)
            show_mask(final_mask, ax)
            show_points(input_points, input_labels, ax)
            ax.set_title(f"{fname}\n{len(clicked_points)} Punkt(e) gesetzt (grün=+, lila=-)\nEnter = speichern, nächstes Bild")
            plt.axis('off')
            fig.canvas.draw()

    def on_key(event):
        if event.key == "enter":
            plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)
    ax.set_title(f"{fname}\nKlicke ins Bild, um Punkte zu setzen\nEnter = speichern, nächstes Bild")
    plt.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # Jetzt erst Bild und Maske gemeinsam auf 1024x1024 bringen
    img_1024 = resize_and_crop(img, 1024)
    if final_mask is not None:
        mask_arr = np.squeeze((final_mask > 0).astype(np.uint8)) * 255  # Shape (H, W)
        mask_img = Image.fromarray(mask_arr)
        mask_1024 = resize_and_crop(mask_img, 1024)
    else:
        mask_1024 = Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8))

    # Speichern
    img_1024.save(os.path.join(output_dir, "images", os.path.splitext(fname)[0] + ".jpg"), quality=95)
    mask_1024.save(os.path.join(output_dir, "masks", os.path.splitext(fname)[0] + ".png"))

    shutil.rmtree(tmpdir)

print("Fertig! Alle Bilder und Masken gespeichert.")