# 1. Bibliotheken importieren
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor

# 2. Video laden und in Frames umwandeln
# Das Video wird mit OpenCV geöffnet und alle Frames werden als RGB-Bilder in eine Liste geladen.
#video_path = r"demo\data\gallery\01_dog_short.mp4" ## Windoofs

video_path = os.path.join("demo", "data", "gallery", "01_dog_short.mp4")

cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV liest BGR, wir brauchen RGB
    frames.append(frame_rgb)
cap.release()
frames = np.array(frames)
n_frames = len(frames)
print(f"Video geladen: {n_frames} Frames, Größe: {frames[0].shape}")

# 3. SAM2 Video Predictor laden (tiny-Modell)
# Das Modell wird direkt von Hugging Face geladen.
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

# 4. Inferenz-State initialisieren
# Die Frames werden als JPEGs in einem temporären Ordner gespeichert, wie im offiziellen Notebook empfohlen.
import tempfile, os, shutil
tmpdir = tempfile.mkdtemp()
for idx, frame in enumerate(frames):
    Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
inference_state = predictor.init_state(video_path=tmpdir)

# 5. Hilfsfunktionen für Visualisierung
def show_mask(mask, ax, color=None):
    # Zeigt eine Maske auf einer matplotlib-Achse an (Standard: knallrot, 60% Transparenz)
    if color is None:
        color = np.array([1.0, 0.0, 0.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    # Positive Punkte (Label 1) grün, negative (Label 0) lila
    pos_points = coords[np.array(labels) == 1]
    neg_points = coords[np.array(labels) == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

# 6. Interaktive Callback-Funktion für Klicks im ersten Frame
clicked_points = []
clicked_labels = []

def on_click(event):
    # Diese Funktion wird bei jedem Mausklick im Bild aufgerufen.
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
        ax.clear()
        ax.imshow(frames[0])
        if clicked_points:
            # Maske für aktuelle Punkte berechnen und anzeigen
            input_points = np.array(clicked_points, dtype=np.float32)
            input_labels = np.array(clicked_labels, dtype=np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=input_points,
                labels=input_labels,
            )
            show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax)
            show_points(input_points, input_labels, ax)
        ax.set_title(f"{len(clicked_points)} Punkt(e) gesetzt (grün=+, lila=-)\nDrücke Enter zum Starten")
        plt.axis('off')
        fig.canvas.draw()

def on_key(event):
    # Wenn Enter gedrückt wird, wird das Fenster geschlossen und die Propagation gestartet.
    if event.key == "enter":
        plt.close()

# 7. Erstes Frame anzeigen und auf Klicks warten
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(frames[0])
ax.set_title("Klicke ins Bild, um Punkte zu setzen (Enter = Start)")
plt.axis('off')
cid = fig.canvas.mpl_connect('button_press_event', on_click)
kid = fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

# 8. Nach Enter: Maske auf erstem Frame berechnen (nochmal, falls nötig)
if not clicked_points:
    print("Keine Punkte gesetzt. Abbruch.")
    shutil.rmtree(tmpdir)
    exit()

points = np.array(clicked_points, dtype=np.float32)
labels = np.array(clicked_labels, dtype=np.int32)
ann_frame_idx = 0
ann_obj_id = 1  # beliebige ID

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# 9. Masken durchs ganze Video propagieren
print("Propagiere Maske durchs Video ...")
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    # Für jeden Frame werden die Masken gespeichert (pro Objekt-ID)
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# 10. Animation anzeigen: Maske(n) im Video visualisieren
import time
for idx in range(n_frames):
    plt.clf()
    plt.imshow(frames[idx])
    if ann_obj_id in video_segments.get(idx, {}):
        show_mask(video_segments[idx][ann_obj_id], plt.gca())
    plt.title(f"Frame {idx+1}/{n_frames}")
    plt.axis('off')
    plt.pause(0.04)  # ca. 25 FPS
plt.show()

# 11. Aufräumen: Temporären Ordner löschen
shutil.rmtree(tmpdir)