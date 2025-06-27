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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sam2.sam2_video_predictor import SAM2VideoPredictor
####################################################################################################

LOG_DIR = "logs"      
os.makedirs(LOG_DIR, exist_ok=True)


EXPECTED_POTHOLES = {
    ## bilder von meister betram die KEINE potholes sind 
    '1084558642946786.jpg': False,
    '1188829775108914.jpg': False,
    '1285029535699522.jpg': False,
    '1316023229129292.jpg': False,
    '1373675990142727.jpg': False,
    '1392211998201369.jpg': False,
    '1529193104275031.jpg': False,
    '159169470045989.jpg': False,
    '1607397469702849.jpg': False,
    '1640108929745447.jpg': False,
    '1933883046948570.jpg': False,
    '1947782532230929.jpg': False,
    '206307738697485.jpg': False,
    '2086818021517027.jpg': False,
    '2183222621861582.jpg': False,
    '218423267526846.jpg': False,
    '225307656701224.jpg': False, # nicht eindeutig evtl auch true
    '2285155878324613.jpg': False,
    '3407607062790162.jpg': False,
    '3519886001580000.jpg': False,
    '4423489001108853.jpg': False,
    '528958382754914.jpg': False, # nicht eindeutig
    '539037624915774.jpg': False,
    '5835345579927252.jpg': False,
    '588806949941319.jpg': False,
    '600234825378353.jpg': False,
    '600284248638387.jpg': False,
    '609733057343978.jpg': False,
    '609992730688692.jpg': False,
    '630792852391348.jpg': False,
    '6308437819176616.jpg': False,
    '694820872645465.jpg': False,
    '699547931969307.jpg': False,
    '753553316278818.jpg': False,
    '900484341229742.jpg': False,
    '909202196973702.jpg': False,
    '925352545321824.jpg': False, # nicht eindeutig
    '933059714605078.jpg': False,
    '953762002733107.jpg': False,
##### ab hier eindeutige Potholes, nicht mehr alle von meister bertram
    '199931331957826.jpg': True,
    '228555592368313.jpg': True,
    '1207079423611781.jpg': True,
    '763828355049428.jpg': True,
    '740633213277185.jpg': True,
    '296134512063765.jpg': True,
    '293654665647732.jpg': True,
    '131431262278312.jpg': True,
    '1640108929745447.jpg': True,
    '4118924161464361.jpg': True,
    '248550337023183.jpg': True,
    '3929816854006196.jpg': True,
    '1150488025378390.jpg': True,
    '427180291932701.jpg': True,
    '2846149118958684.jpg': True,
    '985985122625097.jpg': True,
    '968626133944738.jpg': True,
    '133763585410017.jpg': True,
    '162504769139920.jpg': True,
    '1267823477007713.jpg': True,
    '528958382754914.jpg': True,
    '515767156244364.jpg': True,
    '961729745740668.jpg': True,
    '1969647903182374.jpg': True,
    '153687263398545.jpg': True,
    '496748668144453.jpg': True,
    '137496905033902.jpg': True,
    '156567849760040.jpg': True,
    '297333548545759.jpg': True,
    '761794412794226.jpg': True
}
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

        "per_image": []        # hier hängen wir später dicts an
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
            return EXPECTED_POTHOLES.get(name_lower, False)
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