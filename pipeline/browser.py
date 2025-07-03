import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class ImageBrowser:
    def __init__(self, frames, segments, names):
        self.frames, self.seg, self.names = frames, segments, names
        self.i, self.n = 0, len(frames)
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        plt.subplots_adjust(bottom=0.2)
        self.img_artist = self.ax.imshow(self.frames[0]); self.mask = None
        self.ax.axis("off"); self._title()

        self.bprev = Button(plt.axes([0.3,0.05,0.1,0.075]), "Zurück")
        self.bnext = Button(plt.axes([0.6,0.05,0.1,0.075]), "Weiter")
        self.bprev.on_clicked(lambda _: self._step(-1))
        self.bnext.on_clicked(lambda _: self._step(+1))
        self._draw_mask(); plt.show()

    def _title(self): self.ax.set_title(f"{self.names[self.i]} ({self.i+1}/{self.n})")
    def _draw_mask(self):
        # Vorherige Maske entfernen
        if self.mask is not None:
            self.mask.remove()
            self.mask = None

        # Gibt es für das aktuelle Bild (Index self.i) eine Maske mit ID 1?
        if 1 in self.seg.get(self.i, {}):
            m = self.seg[self.i][1]      # Maske holen (Shape: (H,W) oder (1,H,W))
            m2d = m.squeeze()            # (1,H,W) → (H,W)
            h, w = m2d.shape

            rgba = np.array([1, 0, 0, 0.6])               # Rot + Alpha
            mask_img = m2d.reshape(h, w, 1) * rgba.reshape(1, 1, -1)
            self.mask = self.ax.imshow(mask_img)

    def _step(self, d):
        if 0 <= self.i+d < self.n:
            self.i += d
            self.img_artist.set_data(self.frames[self.i]); self._title(); self._draw_mask()
            self.fig.canvas.draw_idle()
