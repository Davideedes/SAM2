from pathlib import Path
import numpy as np
from PIL import Image

def load_and_resize(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img.resize(size, Image.LANCZOS))

def save_mask_npz(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(out_path, mask=mask.astype(np.uint8))
