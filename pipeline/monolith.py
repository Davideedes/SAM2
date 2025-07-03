import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile, os, shutil
import sys
from matplotlib.widgets import Button
import optparse
import json
import datetime
import time
import argparse
from pathlib import Path
from sam2.sam2_video_predictor import SAM2VideoPredictor
####################################################################################################
LOG_DIR = "logs"      
os.makedirs(LOG_DIR, exist_ok=True)
config_path = Path(__file__).parent / "ground_truth_config.json"
with open(config_path, "r") as f:
    GROUND_TRUTH = json.load(f)
####################################################################################################
def _new_run_dict(ts, n_train, model_size, seq_folder, masks_folder):
    return {
        "timestamp": ts,
        "model_size": model_size,
        "n_train": n_train,
        "seq_folder": os.path.basename(seq_folder),
        "masks_folder": os.path.basename(masks_folder) if masks_folder else None,

        "expected_total": 0,
        "found_total":    0,

        "true_positives":  0,
        "true_negatives":  0,
        "false_negatives": 0,
        "false_positives": 0,

        "per_image": []    
    }


def run_cross_image_transfer(n_train, model_size, seq_folder, masks_folder = None):
    # >>> Monitoring initialisieren
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    start_time      = time.time()  
    if masks_folder:                         # NEW -------------------------------------------------
        os.makedirs(masks_folder, exist_ok=True)
    run_log = _new_run_dict(ts, n_train, model_size, seq_folder, masks_folder)
    # -------------------------------
    # ---------------- dein Originalcode (unverändert) ----------------
    train_image_names = [
        "Schlagloch1.jpeg","Schlagloch2.jpeg","Schlagloch3.jpeg",
        "Schlagloch4.jpeg","Schlagloch7.jpeg","Schlagloch8.jpeg","Schlagloch11.jpg",
    ]
    train_image_paths = [os.path.join("testbilder", n) for n in train_image_names[:n_train]]

    seq_image_names = sorted([f for f in os.listdir(seq_folder)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    seq_image_paths = [os.path.join(seq_folder, n) for n in seq_image_names]

    train_imgs = [Image.open(p).convert("RGB") for p in train_image_paths]
    min_width = min(i.width for i in train_imgs); min_height = min(i.height for i in train_imgs)
    target_size = (min_width, min_height)
    train_frames = [np.array(img.resize(target_size, Image.LANCZOS)) for img in train_imgs]

    predictor = SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")

    all_frames, all_video_segments = [], {}
    for seq_idx, seq_path in enumerate(seq_image_paths):
        seq_img = Image.open(seq_path).convert("RGB")
        seq_img_resized = np.array(seq_img.resize(target_size, Image.LANCZOS))
        frames = train_frames + [seq_img_resized]

        tmpdir = tempfile.mkdtemp()
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmpdir, f"{idx:05d}.jpg"))
        inference_state = predictor.init_state(video_path=tmpdir)

        for i, name in enumerate(train_image_names[:n_train]):
            maskfile = os.path.join("train_masks", f"train_mask_{os.path.splitext(name)[0]}.npz")
            data = np.load(maskfile); points, labels = data["points"], data["labels"]
            if len(points) > 0:
                predictor.add_new_points_or_box(inference_state, i, 1, points, labels)

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        all_frames.append(seq_img_resized)
        last_idx = len(frames)-1
        all_video_segments[len(all_frames)-1] = video_segments.get(last_idx, {})
        shutil.rmtree(tmpdir)

        def is_expected_pothole(name: str) -> bool:
            name_lower = name.lower()
            return GROUND_TRUTH.get(name_lower, False)
        img_name   = seq_image_names[seq_idx]
        expected = is_expected_pothole(img_name)
        mask       = all_video_segments[seq_idx].get(1, True)
        found      = bool(mask is not True and mask.sum() > 0)
        if expected and found and masks_folder:
            out_name = os.path.splitext(img_name)[0] + ".npz"
            out_path = os.path.join(masks_folder, out_name)
            # wir speichern einfach das Binär-Mask-Array (0/1) – weitere Infos je nach Bedarf
            np.savez(out_path, mask=mask.astype(np.uint8))
            print(f"✅  Maske gespeichert: {out_path}")
        run_log["per_image"].append({
            "img_name": img_name,
            "expected": expected,
            "found":    found,
            "type": ("tp" if  expected and  found else
                    "tn" if not expected and not found else
                    "fn" if  expected and not found else
                    "fp")
        })


        # Gesamt-Zähler updaten
        if expected:  run_log["expected_total"] += 1
        if found:     run_log["found_total"]    += 1

        if expected and found:
            run_log["true_positives"] += 1
        elif not expected and not found:
            run_log["true_negatives"] += 1
        elif expected and not found:
            run_log["false_negatives"] += 1
        elif not expected and found:
            run_log["false_positives"] += 1

        run_log["expected_total"] = run_log["true_positives"] + run_log["false_negatives"]
        run_log["found_total"]    = run_log["true_positives"] + run_log["false_positives"]
        run_log["runtime_seconds"] = round(time.time() - start_time, 3)

    timestamp_str = ts.replace(":", "-").replace("T", "_")
    log_filename = f"Model{model_size}_nTrain{n_train}_{timestamp_str}.json"
    json_path = os.path.join(LOG_DIR, log_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)
    print(f"💾 Log gespeichert unter: {json_path}")

    print(f"💾 Log gespeichert unter: {json_path}")

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










####################################################################################################
# Beispiel-Aufruf ohne argparse (parameter im Skript definiert):
# if __name__ == "__main__":
#     run_cross_image_transfer(
#         n_train=6,  # z.B. 3 Trainingsbilder
#         model_size="tiny",  # tiny, small, base_plus, large
#         seq_folder=os.path.join("seq", "meister_bertram_mit_eindeutigen_potholes")
#     )


## aufruf mit parameter übergeben bei skript aufruf
# python3 erste_anwendung/cross_image_transfer.py \
#        --model_size tiny \
#        --n_train 6 \
#        --seq_folder seq/meister_bertram_mit_eindeutigen_potholes \
#        --masks_folder gespeicherte_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2 cross-image transfer.")
    parser.add_argument("--model_size", type=str, required=True,
                        choices=["tiny", "small", "base_plus", "large"],
                        help="SAM2-Gewichtsvariante.")
    parser.add_argument("--n_train", type=int, required=True,
                        help="Anzahl Trainingsbilder.")
    parser.add_argument("--seq_folder", type=str, required=True,
                        help="Ordner mit den Sequenzbildern.")
    # ---------- NEU ----------
    parser.add_argument("--masks_folder", type=str, default=None,
                        help="Optional: Zielordner, in den True-Positive-Masken (NPZ) geschrieben werden.")
    # -------------------------

    args = parser.parse_args()

    run_cross_image_transfer(
        n_train      = args.n_train,
        model_size   = args.model_size,
        seq_folder   = args.seq_folder,
        masks_folder = args.masks_folder        # NEW
    )