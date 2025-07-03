from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parent
_RES  = _ROOT / "resources"

def load_ground_truth() -> dict[str, bool]:
    with open(_RES / "ground_truth_config.json", "r") as fh:
        return json.load(fh)
    

LOG_DIR = (_ROOT / ".." / "logs").resolve()
LOG_DIR.mkdir(exist_ok=True)
