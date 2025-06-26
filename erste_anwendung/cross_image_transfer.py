import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, os, shutil
from sam2.sam2_video_predictor import SAM2VideoPredictor
from matplotlib.widgets import Button

def run_cross_image_transfer(n_train, model_size):
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

    # Testbilder wie gehabt
    test_image_paths = [
        os.path.join("testbilder", "Schlagloch4.jpeg"),
        # os.path.join("testbilder", "Schlagloch10.jpeg"),
        os.path.join("testbilder", "Schlagloch6.jpeg"),
        os.path.join("testbilder", "Schlagloch8.jpeg"),
        os.path.join("testbilder", "Schlagloch11.jpg"),
        os.path.join("testbilder", "Schlagloch5.jpeg"),
        os.path.join("testbilder", "Schlagloch9.jpeg"),
    ]

    # Alle Bilder laden und auf kleinste Auflösung bringen
    frame_paths = train_image_paths + test_image_paths
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    min_width = min(img.width for img in imgs)
    min_height = min(img.height for img in imgs)
    target_size = (min_width, min_height)
    frames = [np.array(img.resize(target_size, Image.LANCZOS)) for img in imgs]
    n_frames = len(frames)
    print(f"{n_frames} Frames geladen, skaliert auf {target_size}")

    # Predictor laden
    predictor = SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")

    # Frames als JPEGs in temporären Ordner speichern
    tmpdir = tempfile.mkdtemp()
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
    inference_state = predictor.init_state(video_path=tmpdir)

    # Masken für die Trainingsbilder automatisch setzen
    for i, name in enumerate(train_image_names[:n_train]):
        outname = os.path.splitext(name)[0]
        maskfile = os.path.join("train_masks", f"train_mask_{outname}.npz")  # <-- angepasst!
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

    print("Masken gesetzt. Propagiere Masken auf die restlichen Bilder ...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    shutil.rmtree(tmpdir)

    # Optional: Bildbrowser wie gehabt
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

    ImageBrowser(frames, video_segments)

# Beispiel-Aufruf:
if __name__ == "__main__":
                            
    run_cross_image_transfer(n_train=7, # max 7
                             model_size="tiny" # tiny, small, base_plus, large
                             ) 