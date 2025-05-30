import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Modell laden (tiny oder large möglich)
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
# Für das große Modell:
# predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

# Video-State initialisieren (Pfad zu JPEG-Frames-Ordner oder Video)
state = predictor.init_state(video_path=r"demo\data\gallery\01_dog_short.mp4")  # oder video_path="demo/data/gallery/01_dog.mp4"

# Beispiel-Prompts (hier: ein Punkt im ersten Frame, Label 1)
points = np.array([[100, 100]], dtype=np.float32)
labels = np.array([1], dtype=np.int32)
frame_idx = 0
obj_id = 1

# Maske im ersten Frame berechnen
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=state,
    frame_idx=frame_idx,
    obj_id=obj_id,
    points=points,
    labels=labels,
)

# Masken durchs Video propagieren
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
    # out_mask_logits: Maske(n) für diesen Frame
    pass

# --- 1. Video laden und in Frames umwandeln ---
video_path = r"demo\data\gallery\01_dog_short.mp4"
cap = cv2.VideoCapture(video_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
cap.release()
frames = np.array(frames)
n_frames = len(frames)
print(f"Video geladen: {n_frames} Frames, Größe: {frames[0].shape}")

# --- 2. SAM2 Video Predictor laden ---
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

# --- 3. Inferenz-State initialisieren ---
# Wir speichern die Frames als JPEGs im RAM, wie im Notebook empfohlen
import tempfile, os, shutil
tmpdir = tempfile.mkdtemp()
for idx, frame in enumerate(frames):
    Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
inference_state = predictor.init_state(video_path=tmpdir)

# --- 4. Interaktive Punkt-Auswahl auf erstem Frame ---
clicked_points = []
clicked_labels = []

def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([1.0, 0.0, 0.0, 0.6])  # Knallrot
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
        ax.clear()
        ax.imshow(frames[0])
        if clicked_points:
            # Maske für aktuelle Punkte berechnen
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
    if event.key == "enter":
        plt.close()

# Erstes Frame anzeigen und Punkte setzen
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(frames[0])
ax.set_title("Klicke ins Bild, um Punkte zu setzen (Enter = Start)")
plt.axis('off')
cid = fig.canvas.mpl_connect('button_press_event', on_click)
kid = fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

# --- 5. Maske auf erstem Frame berechnen ---
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

# --- 6. Masken durchs ganze Video propagieren ---
print("Propagiere Maske durchs Video ...")
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# --- 7. Animation anzeigen ---
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

# --- 8. Aufräumen ---
shutil.rmtree(tmpdir)