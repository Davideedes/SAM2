import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, os, shutil
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from matplotlib.widgets import Button

def run_cross_image_transfer(model_size):
    train_image_path = "combined_train_image.jpg"

    # Testbilder wie gehabt
    test_image_paths = [
        os.path.join("testbilder", "Schlagloch4.jpeg"),
        os.path.join("testbilder", "Schlagloch6.jpeg"),
        os.path.join("testbilder", "Schlagloch8.jpeg"),
        os.path.join("testbilder", "Schlagloch11.jpg"),
        os.path.join("testbilder", "Schlagloch5.jpeg"),
        os.path.join("testbilder", "Schlagloch9.jpeg"),
    ]

    # Alle Bilder laden und auf kleinste Auflösung bringen
    frame_paths = [train_image_path] + test_image_paths
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    min_width = min(img.width for img in imgs)
    min_height = min(img.height for img in imgs)
    target_size = (min_width, min_height)
    frames = [np.array(img.resize(target_size, Image.LANCZOS)) for img in imgs]
    n_frames = len(frames)
    print(f"{n_frames} Frames geladen, skaliert auf {target_size}")

    # --- Interaktive Maskenwahl mit ImagePredictor ---
    image_np = frames[0]
    predictor_img = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
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

    def show_mask(mask, ax, color=None):
        if color is None:
            color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = np.array(coords)[np.array(labels) == 1]
        neg_points = np.array(coords)[np.array(labels) == 0]
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                       edgecolor='white', linewidth=1.25, label="Positiv")
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*', s=marker_size,
                       edgecolor='white', linewidth=1.25, label="Negativ")

    def redraw():
        ax.clear()
        ax.imshow(image_np)
        for i, mask in enumerate(all_masks):
            show_mask(mask, ax, color=mask_colors[i % len(mask_colors)])
        if current_mask is not None:
            show_mask(current_mask, ax, color=mask_colors[len(all_masks) % len(mask_colors)])
        for i, (pts, lbls) in enumerate(zip(all_points, all_labels)):
            show_points(pts, lbls, ax)
        if clicked_points:
            show_points(np.array(clicked_points), np.array(clicked_labels), ax)
        ax.set_title(
            f"{len(all_masks) + (1 if current_mask is not None else 0)} Masken. "
            "Tab = neue Maske, Enter = speichern und propagieren"
        )
        plt.axis('off')
        fig.canvas.draw()

    def on_click(event):
        nonlocal current_mask
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
            with torch.inference_mode():
                predictor_img.set_image(image_np)
                masks, scores, logits = predictor_img.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
            current_mask = masks[0]
            redraw()

    def on_key(event):
        nonlocal clicked_points, clicked_labels, current_mask
        if event.key == "tab":
            if current_mask is not None:
                all_points.append(np.array(clicked_points))
                all_labels.append(np.array(clicked_labels))
                all_masks.append(current_mask)
            clicked_points = []
            clicked_labels = []
            current_mask = None
            redraw()
        elif event.key == "enter":
            if current_mask is not None:
                all_points.append(np.array(clicked_points))
                all_labels.append(np.array(clicked_labels))
                all_masks.append(current_mask)
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    ax.set_title("Klicke ins Bild, um Punkte zu setzen\nTab = neue Maske, Enter = speichern und propagieren")
    plt.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # --- Ab hier wie gehabt: VideoPredictor für Propagation ---
    print(f"{len(all_masks)} Masken werden propagiert ...")

    predictor = SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")

    tmpdir = tempfile.mkdtemp()
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
    inference_state = predictor.init_state(video_path=tmpdir)

    for i in range(len(all_points)):
        points = all_points[i]
        labels = all_labels[i]
        if len(points) > 0:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,  # Nur das Trainingsbild
                obj_id=i+1,   # Jede Maske bekommt eine eigene obj_id
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

    # Bildbrowser wie gehabt
    ImageBrowser(frames, video_segments, mask_colors)

class ImageBrowser:
    def __init__(self, frames, video_segments, mask_colors):
        self.frames = frames
        self.video_segments = video_segments
        self.mask_colors = mask_colors
        self.idx = 0
        self.n = len(frames)
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2)
        self.img = self.ax.imshow(self.frames[self.idx])
        self.mask_artists = []
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
        # Entferne alte Masken
        for artist in getattr(self, "mask_artists", []):
            artist.remove()
        self.mask_artists = []
        # Zeige alle Masken in unterschiedlichen Farben
        masks_dict = self.video_segments.get(self.idx, {})
        for i, obj_id in enumerate(sorted(masks_dict.keys())):
            mask = masks_dict[obj_id]
            color = self.mask_colors[(obj_id-1) % len(self.mask_colors)]
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            self.mask_artists.append(self.ax.imshow(mask_image))

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

if __name__ == "__main__":
    run_cross_image_transfer(model_size="tiny")