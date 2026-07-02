"""Garment catalog.

Loads every RGBA PNG from static/garments/ at startup. Drop any
transparent-background clothing PNG into that folder and it appears in
the app automatically — no code change needed.
"""

import os

import cv2
import numpy as np

GARMENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static",
    "garments",
)


class GarmentCatalog:
    def __init__(self, directory: str = GARMENT_DIR):
        self.directory = directory
        self._garments: dict[str, np.ndarray] = {}
        self.reload()

    def reload(self):
        self._garments.clear()
        if not os.path.isdir(self.directory):
            return
        for name in sorted(os.listdir(self.directory)):
            if not name.lower().endswith(".png"):
                continue
            path = os.path.join(self.directory, name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.shape[2] == 3:  # no alpha channel -> add opaque alpha
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            key = os.path.splitext(name)[0]
            self._garments[key] = img

    def names(self) -> list[str]:
        return list(self._garments.keys())

    def get(self, name: str) -> np.ndarray | None:
        return self._garments.get(name)

    def first(self) -> str | None:
        names = self.names()
        return names[0] if names else None
