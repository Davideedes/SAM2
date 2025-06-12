"""
sam2_module.py
~~~~~~~~~~~~~~
Interaktive Segmentierung mit SAM 2 und Hilfsroutinen,
um eine Binärmaske zu erzeugen und auf das Originalbild anzuwenden.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Sam2Segmenter:
    def __init__(self,
                 model_name: str = "facebook/sam2-hiera-tiny",
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        self._clicked_points: list[list[int]] = []
        self._clicked_labels: list[int] = []
        self._final_mask: np.ndarray | None = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _show_mask(mask: np.ndarray, ax, color=(1.0, 0.0, 0.0, 0.6)):
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def _show_points(coords, labels, ax, marker_size=375):
        pos = coords[np.array(labels) == 1]
        neg = coords[np.array(labels) == 0]
        if len(pos):
            ax.scatter(pos[:, 0], pos[:, 1], color="lime", marker="*", s=marker_size,
                       edgecolor="white", linewidth=1.25, label="Positiv")
        if len(neg):
            ax.scatter(neg[:, 0], neg[:, 1], color="#a020f0", marker="*", s=marker_size,
                       edgecolor="white", linewidth=1.25, label="Negativ")

    # ------------------------------------------------------------------ public
    def segment_interactive(self, image_path: str):
        """
        Öffnet ein Matplotlib-Fenster, lässt Klicks zu und
        gibt (maske_np, image_np) zurück, sobald ENTER gedrückt wurde.
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Callback-Funktionen ---------------------------------------------------
        def on_click(event):
            if event.inaxes is None:
                return
            x, y = int(event.xdata), int(event.ydata)
            if event.button == 1:
                self._clicked_points.append([x, y]); self._clicked_labels.append(1)
            elif event.button == 3:
                self._clicked_points.append([x, y]); self._clicked_labels.append(0)
            else:
                return

            pts = np.array(self._clicked_points)
            lbl = np.array(self._clicked_labels)

            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16
                                                        if self.device == "cuda" else torch.float32):
                self.predictor.set_image(image_np)
                masks, *_ = self.predictor.predict(
                    point_coords=pts,
                    point_labels=lbl,
                    multimask_output=True,
                )
            self._final_mask = masks[0]

            ax.clear()
            ax.imshow(image_np)
            self._show_mask(self._final_mask, ax)
            self._show_points(pts, lbl, ax)
            ax.set_title(f"{len(pts)} Punkt(e) • ENTER = fertig")
            ax.axis("off")
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key == "enter":
                plt.close()

        # Matplotlib-Interaktion ----------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_np)
        ax.set_title("Klicke ins Bild • ENTER = fertig")
        ax.axis("off")
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        if self._final_mask is None:
            raise RuntimeError("Es wurde keine Maske erzeugt.")

        return self._final_mask.astype(bool), image_np

    # ------------------------------------------------------------------ utils
    @staticmethod
    def apply_mask(image_np: np.ndarray, mask_bool: np.ndarray) -> Image.Image:
        """Setzt alle Hintergrundpixel auf Schwarz und gibt PIL-Image zurück."""
        out = np.zeros_like(image_np)
        out[mask_bool] = image_np[mask_bool]
        return Image.fromarray(out)

    @staticmethod
    def apply_mask_with_context(image_np: np.ndarray, mask_bool: np.ndarray) -> Image.Image:
        """Hebt nur das Objekt hervor, dunkelt den Rest ab"""
        # Kopie und Umrechnung
        out = image_np.copy()
        out[~mask_bool] = (out[~mask_bool] * 0.2).astype(np.uint8)  # Abdunkeln
        return Image.fromarray(out)

    @staticmethod
    def save_mask(mask_bool: np.ndarray, path: str):
        from PIL import Image as _PIL
        _PIL.fromarray(mask_bool.astype(np.uint8) * 255).save(path)


    # ------------------------------------------------------------------ utils  (unten anhängen)
    # ----------  Bounding-Box aus Maske  ----------
    @staticmethod
    def mask_to_bbox(mask_bool: np.ndarray, padding: float = 0.10) -> tuple[int, int, int, int]:
        """
        Liefert (x_min, y_min, x_max, y_max)   –  inkl. optionalem Rand (padding in %)
        """
        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            raise ValueError("Leere Maske – BBox kann nicht berechnet werden")

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # --- kontextueller Rand ------------------------------------------------
        h, w = mask_bool.shape
        pad_x = int((x_max - x_min + 1) * padding)
        pad_y = int((y_max - y_min + 1) * padding)
        x_min = max(x_min - pad_x, 0)
        y_min = max(y_min - pad_y, 0)
        x_max = min(x_max + pad_x, w - 1)
        y_max = min(y_max + pad_y, h - 1)
        return x_min, y_min, x_max, y_max

    # ----------  Bild auf Bounding-Box zuschneiden  ----------
    @staticmethod
    def crop_bbox(image_np: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
        x_min, y_min, x_max, y_max = bbox
        return Image.fromarray(image_np[y_min : y_max + 1, x_min : x_max + 1])

    # ----------  BBox ins Bild zeichnen (Debug/Overlay) ----------
    @staticmethod
    def draw_bbox_on_image(image: Image.Image,
                           bbox: tuple[int, int, int, int],
                           color: str = "red",
                           width: int = 5) -> Image.Image:
        from PIL import ImageDraw
        out = image.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle(bbox, outline=color, width=width)
        return out

