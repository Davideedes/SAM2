from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parent
RES  = _ROOT / "resources"
RESOURCE_DIR = Path(__file__).parent / "resources"
TRAIN_DIR = RESOURCE_DIR / "input_pictures"
TRAIN_NPZ_MASK_DIR = RESOURCE_DIR / "input_pictures_npz_masks_ground_truth"
LOG_DIR = (_ROOT / "logs").resolve()
LOG_DIR.mkdir(exist_ok=True)


def load_ground_truth() -> dict[str, bool]:
    with open(RES / "ground_truth_config.json", "r") as fh:
        return json.load(fh)
    
def load_input_sample_pictures() -> list[str]:
    with open(RES / "input_pictures_config.json") as fh:
        return json.load(fh)
    

