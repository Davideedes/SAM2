import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, os, shutil
from sam2.sam2_video_predictor import SAM2VideoPredictor
from matplotlib.widgets import Button

def run_cross_image_transfer(model_size, seq_folder):
    # Die ersten 4 Trainingsbilder und Masken
    train_image_names = [
        "Schlagloch1.jpeg",
        "Schlagloch2.jpeg",
        "Schlagloch3.jpeg",
        "Schlagloch4.jpeg",
    ]
    mask_dir = "train_masks"
    img_dir = "testbilder"

    # Bilder und Masken laden
    imgs = [Image.open(os.path.join(img_dir, name)).convert("RGB") for name in train_image_names]
    min_width = min(img.width for img in imgs)
    min_height = min(img.height for img in imgs)
    imgs = [img.resize((min_width, min_height), Image.LANCZOS) for img in imgs]

    # Maskenpunkte und Labels laden
    all_points = []
    all_labels = []
    for name in train_image_names:
        outname = os.path.splitext(name)[0]
        data = np.load(os.path.join(mask_dir, f"train_mask_{outname}.npz"))
        all_points.append(data["points"])
        all_labels.append(data["labels"])

    # 2x2-Grid erstellen
    combined_img = Image.new("RGB", (2 * min_width, 2 * min_height))
    offsets = [(0, 0), (min_width, 0), (0, min_height), (min_width, min_height)]
    combined_points = []
    combined_labels = []
    obj_ids = []

    for i, (img, points, labels) in enumerate(zip(imgs, all_points, all_labels)):
        x_off, y_off = offsets[i]
        combined_img.paste(img, (x_off, y_off))
        # Punkte verschieben
        if len(points) > 0:
            shifted = points + np.array([x_off, y_off])
            combined_points.append(shifted)
            combined_labels.append(labels)
            obj_ids.append(np.full(len(points), i+1))  # obj_id: 1,2,3,4

    # Alle Punkte, Labels und obj_ids zusammenführen
    if combined_points:
        combined_points = np.concatenate(combined_points, axis=0)
        combined_labels = np.concatenate(combined_labels, axis=0)
        obj_ids = np.concatenate(obj_ids, axis=0)
    else:
        combined_points = np.zeros((0, 2))
        combined_labels = np.zeros((0,))
        obj_ids = np.zeros((0,))

    # Kombiniertes Bild als Trainingsbild
    combined_img_np = np.array(combined_img)

    # Sequenzbilder laden und auf gleiche Größe bringen
    seq_image_names = sorted([f for f in os.listdir(seq_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    seq_image_paths = [os.path.join(seq_folder, name) for name in seq_image_names]
    seq_imgs = [Image.open(p).convert("RGB").resize((2*min_width, 2*min_height), Image.LANCZOS) for p in seq_image_paths]
    seq_frames = [np.array(img) for img in seq_imgs]

    # Für jedes Sequenzbild ein eigenes "Video" (Trainingsbild + Sequenzbild)
    all_results = []
    mask_colors = [
        np.array([1.0, 0.0, 0.0, 0.5]),   # Rot
        np.array([0.0, 1.0, 0.0, 0.5]),   # Grün
        np.array([0.0, 0.0, 1.0, 0.5]),   # Blau
        np.array([1.0, 1.0, 0.0, 0.5]),   # Gelb
    ]

    for seq_idx, (seq_img, seq_name) in enumerate(zip(seq_frames, seq_image_names)):
        frames = [combined_img_np, seq_img]
        predictor = SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")

        tmpdir = tempfile.mkdtemp()
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
        inference_state = predictor.init_state(video_path=tmpdir)

        # Masken für das Trainingsbild setzen (je obj_id)
        for obj_id in range(1, 5):
            mask_points = combined_points[obj_ids == obj_id]
            mask_labels = combined_labels[obj_ids == obj_id]
            if len(mask_points) > 0:
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=mask_points,
                    labels=mask_labels,
                )

        # Propagiere Masken auf das Sequenzbild
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        shutil.rmtree(tmpdir)
        all_results.append((frames, video_segments, seq_name))

    # Bildbrowser für alle Paare (Trainingsbild + Sequenzbild)
    class ImageBrowser:
        def __init__(self, all_results):
            self.all_results = all_results
            self.idx = 0
            self.n = len(all_results)
            self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 8))
            plt.subplots_adjust(bottom=0.2)
            self.imgs = [self.axs[0].imshow(self.all_results[self.idx][0][0]),  # Trainingsbild
                         self.axs[1].imshow(self.all_results[self.idx][0][1])]  # Sequenzbild
            self.mask_artists = [[], []]
            self.axs[0].set_title("Trainingsbild (kombiniert)")
            self.axs[1].set_title(f"Sequenzbild: {self.all_results[self.idx][2]}")
            for ax in self.axs:
                ax.axis('off')
            self.add_mask()

            axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
            axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
            self.bnext = Button(axnext, 'Weiter')
            self.bprev = Button(axprev, 'Zurück')
            self.bnext.on_clicked(self.next)
            self.bprev.on_clicked(self.prev)
            plt.show()

        def add_mask(self):
            # Entferne alte Masken
            for mask_list in self.mask_artists:
                for artist in mask_list:
                    artist.remove()
            self.mask_artists = [[], []]
            # Trainingsbild-Masken
            masks_dict = self.all_results[self.idx][1].get(0, {})
            for i, obj_id in enumerate(sorted(masks_dict.keys())):
                mask = masks_dict[obj_id]
                color = mask_colors[(obj_id-1) % len(mask_colors)]
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                self.mask_artists[0].append(self.axs[0].imshow(mask_image))
            # Sequenzbild-Masken
            masks_dict = self.all_results[self.idx][1].get(1, {})
            for i, obj_id in enumerate(sorted(masks_dict.keys())):
                mask = masks_dict[obj_id]
                color = mask_colors[(obj_id-1) % len(mask_colors)]
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                self.mask_artists[1].append(self.axs[1].imshow(mask_image))

        def update(self):
            self.imgs[0].set_data(self.all_results[self.idx][0][0])
            self.imgs[1].set_data(self.all_results[self.idx][0][1])
            self.axs[1].set_title(f"Sequenzbild: {self.all_results[self.idx][2]}")
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
    print(f"Verarbeitete Paare: {len(all_results)}")
    ImageBrowser(all_results)

# Beispiel-Aufruf:
if __name__ == "__main__":
    run_cross_image_transfer(
        model_size="tiny",  # tiny, small, base_plus, large
        seq_folder="seq/seq_meister_bertram_strasse"  # <--- hier deinen Sequenzordner angeben!
    )