import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, os, shutil
from sam2.sam2_video_predictor import SAM2VideoPredictor
from matplotlib.widgets import Button

def run_cross_image_transfer(n_train, model_size, seq_folder):
    # Trainingsbilder und zugehörige Masken
    train_image_names = [
        "Schlagloch1.jpeg",
        "Schlagloch2.jpeg",
        "Schlagloch3.jpeg",
        "Schlagloch4.jpeg",
        "Schlagloch7.jpeg",
        "Schlagloch8.jpeg",
        "Schlagloch11.jpg",
    ]
    train_image_paths = [os.path.join("testbilder", name) for name in train_image_names[:n_train]]

    # Sequenzbilder aus dem angegebenen Ordner laden (sortiert)
    seq_image_names = sorted([f for f in os.listdir(seq_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    seq_image_paths = [os.path.join(seq_folder, name) for name in seq_image_names]

    # Alle Trainingsbilder laden (nur einmal)
    train_imgs = [Image.open(p).convert("RGB") for p in train_image_paths]
    min_width = min(img.width for img in train_imgs)
    min_height = min(img.height for img in train_imgs)
    target_size = (min_width, min_height)
    train_frames = [np.array(img.resize(target_size, Image.LANCZOS)) for img in train_imgs]

    # Predictor laden
    predictor = SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")

    all_frames = []
    all_video_segments = {}

    # Für jedes Bild in der Sequenz: Trainingsbilder + 1 Sequenzbild gemeinsam verarbeiten
    for seq_idx, seq_path in enumerate(seq_image_paths):
        # Lade und skaliere das aktuelle Sequenzbild
        seq_img = Image.open(seq_path).convert("RGB")
        seq_img_resized = np.array(seq_img.resize(target_size, Image.LANCZOS))
        frames = train_frames + [seq_img_resized]

        # Frames als JPEGs in temporären Ordner speichern
        tmpdir = tempfile.mkdtemp()
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
        inference_state = predictor.init_state(video_path=tmpdir)

        # Masken für die Trainingsbilder automatisch setzen
        for i, name in enumerate(train_image_names[:n_train]):
            outname = os.path.splitext(name)[0]
            maskfile = os.path.join("train_masks", f"train_mask_{outname}.npz")
            if not os.path.exists(maskfile):
                raise FileNotFoundError(f"Maskendatei {maskfile} nicht gefunden!")
            data = np.load(maskfile)
            points = data["points"]
            labels = data["labels"]
            if len(points) > 0:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )

        # Propagiere Maske auf das aktuelle Sequenzbild
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Nur das Sequenzbild und seine Maske merken (trainingsbilder werden nicht erneut angezeigt)
        all_frames.append(seq_img_resized)
        # Die Maske für das letzte Bild (Sequenzbild) extrahieren
        last_idx = len(frames) - 1
        all_video_segments[len(all_frames)-1] = video_segments.get(last_idx, {})

        shutil.rmtree(tmpdir)

    # Bildbrowser für alle Sequenzbilder (Trainingsbilder werden nicht angezeigt)
    class ImageBrowser:
        def __init__(self, frames, video_segments, seq_image_names):
            self.frames = frames
            self.video_segments = video_segments
            self.seq_image_names = seq_image_names
            self.idx = 0
            self.n = len(frames)
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            plt.subplots_adjust(bottom=0.2)
            self.img = self.ax.imshow(self.frames[self.idx])
            self.mask_artist = None
            self.ax.set_title(f"{self.seq_image_names[self.idx]} ({self.idx+1}/{self.n})")
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
            self.ax.set_title(f"{self.seq_image_names[self.idx]} ({self.idx+1}/{self.n})")
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
    print(f"Gefundene Sequenzbilder: {len(seq_image_names)}")
    print(f"Verarbeitete Frames: {len(all_frames)}")
    ImageBrowser(all_frames, all_video_segments, seq_image_names)

# Beispiel-Aufruf:
if __name__ == "__main__":
    run_cross_image_transfer(
        n_train=3,  # z.B. 3 Trainingsbilder
        model_size="tiny",  # tiny, small, base_plus, large
        seq_folder="seq\seq_meister_bertram_strasse"  # <--- hier deinen Sequenzordner angeben!
    )