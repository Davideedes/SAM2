from __future__ import annotations
from pathlib import Path
import time, datetime, json, tempfile, os, shutil
import numpy as np
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from .config import load_ground_truth, LOG_DIR
from .vision import load_and_resize, save_mask_npz

GROUND_TRUTH = load_ground_truth()



def _new_run_dict(**kwargs) -> dict:
    return {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "model_size":  kwargs["model_size"],
        "n_train":     kwargs["n_train"],
        "seq_folder":  Path(kwargs["seq_folder"]).name,
        "masks_folder":Path(kwargs["masks_folder"]).name if kwargs["masks_folder"] else None,
        "expected_total": 0, "found_total": 0,
        "true_positives":0,"true_negatives":0,"false_negatives":0,"false_positives":0,
        "per_image": [], "runtime_seconds": 0.0,
    }

def run(n_train:int, model_size:str, seq_folder:str|Path, masks_folder:str|Path|None=None):
    start = time.time()
    seq_folder  = Path(seq_folder)
    masks_folder= Path(masks_folder) if masks_folder else None
    run_log     = _new_run_dict(n_train=n_train, model_size=model_size,
                                seq_folder=seq_folder, masks_folder=masks_folder)

    train_imgs  = [Image.open(Path("testbilder")/n).convert("RGB")
                   for n in ("Schlagloch1.jpeg","Schlagloch2.jpeg","Schlagloch3.jpeg",
                              "Schlagloch4.jpeg","Schlagloch7.jpeg","Schlagloch8.jpeg",
                              "Schlagloch11.jpg")][:n_train]
    tgt_size    = min(i.width for i in train_imgs), min(i.height for i in train_imgs)
    train_np    = [np.array(i.resize(tgt_size, Image.LANCZOS)) for i in train_imgs]

    predictor = SAM2VideoPredictor.from_pretrained(f"facebook/sam2-hiera-{model_size}")

    seq_paths   = sorted(p for p in seq_folder.iterdir()
                         if p.suffix.lower() in {".jpg",".jpeg",".png"})

    all_frames, all_segments = [], {}
    for idx, path in enumerate(seq_paths):
        frames = train_np + [load_and_resize(path, tgt_size)]
        with tempfile.TemporaryDirectory() as tmp:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(Path(tmp)/f"{i:05d}.jpg")
            state = predictor.init_state(video_path=tmp)
            _add_click_points(predictor, state, n_train)   # predictor zusätzlich übergeben

            segments = _propagate(predictor, state)
        all_frames.append(frames[-1])
        all_segments[len(all_frames)-1] = segments.get(len(frames)-1, {})

        _update_metrics(run_log, path.name, segments.get(len(frames)-1, {}), masks_folder)

    run_log["runtime_seconds"] = round(time.time()-start, 3)
    _write_log(run_log)
    return all_frames, all_segments, [p.name for p in seq_paths]

# --- Helfer ----------------------------------------------------
def _add_click_points(predictor, state, n_click_imgs:int):

    import numpy as np, os
    for i, name in enumerate(("Schlagloch1.jpeg","Schlagloch2.jpeg","Schlagloch3.jpeg",
                              "Schlagloch4.jpeg","Schlagloch7.jpeg","Schlagloch8.jpeg",
                              "Schlagloch11.jpg")[:n_click_imgs]):
        data = np.load(Path("training_pictures_masks")/f"train_mask_{os.path.splitext(name)[0]}.npz")
        if data["points"].size:
            predictor.add_new_points_or_box(state, i, 1, data["points"], data["labels"])


def _propagate(predictor, state):
    out = {}
    for f_idx, ids, logits in predictor.propagate_in_video(state):
        out[f_idx] = {oid:(logits[i]>0).cpu().numpy() for i,oid in enumerate(ids)}
    return out

def _update_metrics(log:dict, img_name:str, segs:dict, masks_folder:Path|None):
    exp = GROUND_TRUTH.get(img_name.lower(), False)
    found = bool(1 in segs and segs[1].sum() > 0)

    if exp and found and masks_folder:
        save_mask_npz(segs[1], masks_folder/f"{Path(img_name).stem}.npz")

    log["per_image"].append({"img_name":img_name,"expected":exp,"found":found,
                             "type":("tp" if exp and found else
                                     "tn" if not exp and not found else
                                     "fn" if exp and not found else "fp")})
    if exp:   log["expected_total"] += 1
    if found: log["found_total"]    += 1
    if   exp and found:      log["true_positives"]  += 1
    elif exp and not found:  log["false_negatives"] += 1
    elif not exp and found:  log["false_positives"] += 1
    else:                    log["true_negatives"]  += 1

def _write_log(log:dict):
    ts = log["timestamp"].replace(":","-").replace("T","_")
    name = f"Model{log['model_size']}_nTrain{log['n_train']}_{ts}.json"
    with open(LOG_DIR/name,"w",encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print("💾 Log gespeichert:", LOG_DIR/name)
