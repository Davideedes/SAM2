import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, shutil
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.sam2_video_predictor import SAM2VideoPredictor

# Quelle: alle .jpg aus diesem Ordner
SRC_DIR = Path("pipeline/resources/sequence_to_test_2")
# Ziel: Ground-Truth-ähnliche NPZ-Masken (Key 'mask')
MASK_DIR = Path("pipeline/resources/sequence_to_test_2_npz_masks_ground_truth")
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Wenn True, werden auch leere Masken gespeichert (nicht empfohlen).
SAVE_EMPTY = False

image_paths = sorted(SRC_DIR.glob("*.jpg"))
if not image_paths:
    raise SystemExit(f"Keine .jpg in {SRC_DIR} gefunden.")

# Einmal SAM2 laden
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([1.0, 0.0, 0.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    coords = np.array(coords, dtype=float) if len(coords) else np.zeros((0, 2))
    labels = np.array(labels, dtype=int) if len(labels) else np.zeros((0,), dtype=int)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='#a020f0', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)

def process_one_image(img_path: Path):
    """Interaktiv Punkte klicken, Maske erzeugen & als NPZ (key 'mask') speichern (nur wenn nicht leer)."""
    stem = img_path.stem
    frame = np.array(Image.open(img_path).convert("RGB"))

    # temp-„Video“ mit einem Frame
    tmpdir = tempfile.mkdtemp()
    try:
        Image.fromarray(frame).save(Path(tmpdir) / "00000.jpg")
        state = predictor.init_state(video_path=tmpdir)

        clicked_points, clicked_labels = [], []
        box = {"mask": None}   # Container für die letzte Maske (bool HxW)

        def on_click(event):
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            x, y = int(event.xdata), int(event.ydata)
            if event.button == 1:
                clicked_points.append([x, y]); clicked_labels.append(1)  # positiver Klick
            elif event.button == 3:
                clicked_points.append([x, y]); clicked_labels.append(0)  # negativer Klick
            else:
                return

            ax.clear()
            ax.imshow(frame)
            if clicked_points:
                input_points = np.array(clicked_points, dtype=np.float32)
                input_labels = np.array(clicked_labels, dtype=np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,  # ein Objekt
                    points=input_points,
                    labels=input_labels,
                )
                out_obj_ids = list(map(int, out_obj_ids))
                idx_in_batch = out_obj_ids.index(1) if 1 in out_obj_ids else 0

                m = (out_mask_logits[idx_in_batch] > 0).squeeze().detach().cpu().numpy().astype(bool)
                if m.ndim != 2:
                    m = np.squeeze(m)
                box["mask"] = m

                show_mask(m, ax)
                show_points(input_points, input_labels, ax)

            ax.set_title(f"{stem}: {len(clicked_points)} Punkt(e) (grün=+, lila=-) — Enter = speichern & nächstes Bild")
            ax.axis('off')
            fig.canvas.draw()

        def on_key(event):
            if event.key == "enter":
                plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)
        ax.set_title(f"Klicke in {stem} (Enter = speichern & nächstes Bild)")
        ax.axis('off')
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        # Speichern – nur wenn Maske existiert & nicht leer, außer SAVE_EMPTY=True
        out_npz = MASK_DIR / f"train_mask_{stem}.npz"
        if box["mask"] is None or box["mask"].sum() == 0:
            if SAVE_EMPTY:
                h, w = frame.shape[:2]
                empty = np.zeros((h, w), dtype=bool)
                np.savez(out_npz, mask=empty)
                print(f"ℹ️ Leere Maske gespeichert (SAVE_EMPTY=True): {out_npz}")
            else:
                print(f"⚠️ Keine gültige Maske → NICHT gespeichert: {stem}")
        else:
            np.savez(out_npz, mask=box["mask"].astype(np.uint8))  # Key 'mask' → eval-kompatibel
            print(f"💾 Maske gespeichert: {out_npz}")

        # Optional zusätzlich als PNG:
        # Image.fromarray((box['mask'].astype(np.uint8)*255)).save(MASK_DIR / f"train_mask_{stem}.png")

    finally:
        shutil.rmtree(tmpdir)

# alle Bilder abarbeiten
for p in image_paths:
    process_one_image(p)

print("✅ Alle Bilder verarbeitet.")
