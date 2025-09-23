import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, os, shutil
from sam2.sam2_video_predictor import SAM2VideoPredictor
from matplotlib.widgets import Button

# 1. Bilder als Frames laden und auf kleinste Auflösung bringen
frame_paths = [
    # 'Trainings'Bilder 
    os.path.join("testbilder", "Schlagloch8.jpeg"),
    os.path.join("testbilder", "Schlagloch11.jpg"),
    os.path.join("testbilder", "Schlagloch10.jpeg"),
    os.path.join("testbilder", "Schlagloch6.jpeg"),
    # Bilder zur klassifizierung
    os.path.join("testbilder", "Schlagloch1.jpeg"),
    os.path.join("testbilder", "Schlagloch12.jpg"),
    os.path.join("testbilder", "Schlagloch2.jpeg"),
    os.path.join("testbilder", "Schlagloch3.jpeg"),
    os.path.join("testbilder", "Schlagloch7.jpeg"),
    os.path.join("testbilder", "Schlagloch4.jpeg"),
    # os.path.join("testbilder", "keinSchlagloch1.jpeg"),
    # os.path.join("testbilder", "keinSchlagloch2.jpeg"),
    # os.path.join("testbilder", "keinSchlagloch3.jpeg"),
    # os.path.join("testbilder", "keinSchlagloch4.jpeg"),
    # os.path.join("testbilder", "keinSchlagloch5.jpeg"),
    # os.path.join("testbilder", "keinSchlagloch6.jpeg"),
    # Edgecases 
    os.path.join("testbilder", "Schlagloch5.jpeg"),
    os.path.join("testbilder", "Schlagloch9.jpeg"),
]

# Lade alle Bilder und ermittle die kleinste Auflösung
imgs = [Image.open(p).convert("RGB") for p in frame_paths]
min_width = min(img.width for img in imgs)
min_height = min(img.height for img in imgs)
target_size = (min_width, min_height)

# Skaliere alle Bilder auf die kleinste Auflösung
frames = [np.array(img.resize(target_size, Image.LANCZOS)) for img in imgs]
n_frames = len(frames)
print(f"{n_frames} Frames geladen, skaliert auf {target_size}")

# 2. SAM2 Video Predictor laden
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

# 3. Frames als JPEGs in temporären Ordner speichern
tmpdir = tempfile.mkdtemp()
for idx, frame in enumerate(frames):
    Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
inference_state = predictor.init_state(video_path=tmpdir)

# 4. Hilfsfunktionen für Visualisierung
def show_mask(mask, ax, color=None):
    if color is None:
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

# 5. Interaktive Punktwahl für die ersten drei Bilder
clicked_points_per_frame = []
clicked_labels_per_frame = []

for frame_idx in range(4):
    clicked_points = []
    clicked_labels = []

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
            ax.imshow(frames[frame_idx])
            if clicked_points:
                input_points = np.array(clicked_points, dtype=np.float32)
                input_labels = np.array(clicked_labels, dtype=np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    points=input_points,
                    labels=input_labels,
                )
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax)
                show_points(input_points, input_labels, ax)
            ax.set_title(f"Bild {frame_idx+1}: {len(clicked_points)} Punkt(e) gesetzt (grün=+, lila=-)\nDrücke Enter für nächstes Bild")
            plt.axis('off')
            fig.canvas.draw()

    def on_key(event):
        if event.key == "enter":
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frames[frame_idx])
    ax.set_title(f"Klicke ins Bild {frame_idx+1}, um Punkte zu setzen (Enter = Weiter)")
    plt.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    clicked_points_per_frame.append(clicked_points)
    clicked_labels_per_frame.append(clicked_labels)

# 6. Nach Interaktion: Masken für die ersten drei Frames setzen
for frame_idx in range(0):  # NOTE: Das ist die Anzahl der Frames die am Anfang maskiert werden
    if clicked_points_per_frame[frame_idx]:
        points = np.array(clicked_points_per_frame[frame_idx], dtype=np.float32)
        labels = np.array(clicked_labels_per_frame[frame_idx], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )

print("Propagiere Maske auf das 4. Bild ...")
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }



shutil.rmtree(tmpdir)

class ImageBrowser:
    def __init__(self, frames, video_segments):
        self.frames = frames
        self.video_segments = video_segments
        self.idx = 0
        self.n = len(frames)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2)
        self.img = self.ax.imshow(self.frames[self.idx])
        self.mask_artist = None
        self.ax.set_title(f"Bild {self.idx+1}/{self.n}")
        self.ax.axis('off')
        self.add_mask()

        axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Weiter')
        self.bprev = Button(axprev, 'Zurück')
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        plt.show()

    def add_mask(self):
        if self.mask_artist:
            self.mask_artist.remove()
            self.mask_artist = None
        if 1 in self.video_segments.get(self.idx, {}):
            mask = self.video_segments[self.idx][1]
            color = np.array([1.0, 0.0, 0.0, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            self.mask_artist = self.ax.imshow(mask_image)

    def update(self):
        self.img.set_data(self.frames[self.idx])
        self.ax.set_title(f"Bild {self.idx+1}/{self.n}")
        self.add_mask()
        self.fig.canvas.draw_idle()

    def next(self, event):
        if self.idx < self.n - 1:
            self.idx += 1
            self.update()

    def prev(self, event):
        if self.idx > 0:
            self.idx -= 1
            self.update()

# Am Ende deines Skripts:
ImageBrowser(frames, video_segments)