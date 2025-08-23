import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.sam2_video_predictor import SAM2VideoPredictor


# # Trainingsbilder
# Zielordner für Masken
# mask_dir = "pipeline/resources/masks_from_input_pictures_npz_format"
# train_image_names = [
#   "3360951680713425.jpg",
#   "3929816854006196.jpg",
#   "1805037676897412.jpg",
#   "1647869979311320.jpg",
#   "1128327637671443.jpg",
#   "1168719240306654.jpg",
#   "1215669212538799.jpg"
# ]
# train_image_paths = [os.path.join("pipeline/resources/input_pictures", name) for name in train_image_names]

## Sequenzbilder
mask_dir = "pipeline/resources/sequence_to_test_1_npz_masks_ground_truth"
# train_image_names = [
# "1128327637671443",
# "1150488025378390",
# "1168719240306654",
# "1241877959600094",
# "135253878628490",
# "1432956667044194",
# "155105023160236",
# "1692371564291963",
# "185358320106504",
# "185951710055640",
# "201820665115791",
# "204493437983039",
# "207806254320178",
# "2092280197578815",
# "264710655348649",
# "275010177638445",
# "283829646791470",
# "292432549038685",
# "303634461227577",
# "309200677408951",
# "316571519817000",
# "459701678441749",
# "472290454059251",
# "483295859627730",
# "490280405617271",
# "508265580212867",
# "515495122827888",
# "515634443132461",
# "526584771844886",
# "532539517764142",
# "561025455062364",
# "583092922616432",
# "756251151697650",
# "770837453577013",
# "801126600518934",
# "811907083054696",
# "828536674427803",
# "845409816378785",
# "852574285611373",
# "926680828155459",
# "968298413979608"
# ]
# train_image_paths = [os.path.join("pipeline/resources/sequence_to_test_1", f"{name}.jpg") for name in train_image_names]

train_image_names = [

"358071033113221"
]
train_image_paths = [os.path.join("pipeline/resources/2_neue_bilder", f"{name}.jpg") for name in train_image_names]
# Bilder laden und auf kleinste Auflösung bringen
imgs = [Image.open(p).convert("RGB") for p in train_image_paths]
min_width = min(img.width for img in imgs)
min_height = min(img.height for img in imgs)
target_size = (min_width, min_height)
frames = [np.array(img.resize(target_size, Image.LANCZOS)) for img in imgs]

# Predictor laden
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

# Bilder als JPEGs in temporären Ordner speichern
tmpdir = tempfile.mkdtemp()
for idx, frame in enumerate(frames):
    Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
inference_state = predictor.init_state(video_path=tmpdir)

def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([1.0, 0.0, 0.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = np.array(coords)[np.array(labels) == 1]
    neg_points = np.array(coords)[np.array(labels) == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*', s=marker_size,
                   edgecolor='white', linewidth=1.25)

for frame_idx in range(len(frames)):
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
            ax.set_title(f"{train_image_names[frame_idx]}: {len(clicked_points)} Punkt(e) gesetzt (grün=+, lila=-)\nEnter = Weiter")
            plt.axis('off')
            fig.canvas.draw()

    def on_key(event):
        if event.key == "enter":
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frames[frame_idx])
    ax.set_title(f"Klicke ins Bild {train_image_names[frame_idx]}, um Punkte zu setzen (Enter = Weiter)")
    plt.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # Speichern der Punkte und Labels im Zielordner
    outname = os.path.splitext(train_image_names[frame_idx])[0]
    np.savez(os.path.join(mask_dir, f"train_mask_{outname}.npz"),
             points=np.array(clicked_points), labels=np.array(clicked_labels))
    print(f"Maske für {train_image_names[frame_idx]} gespeichert in {mask_dir}/.")

shutil.rmtree(tmpdir)
print("Alle Masken gespeichert.")