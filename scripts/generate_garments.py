"""Generate sample garment PNGs (transparent background) for the demo.

Creates shaded t-shirts and a hoodie in static/garments/. These are
procedural placeholders — for maximum realism, replace them with photos
of real garments cut out onto transparent backgrounds (see README).

Run from the project root:  python scripts/generate_garments.py
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static",
    "garments",
)

W, H = 800, 900  # canvas; garment fills most of it


def _tshirt_mask() -> Image.Image:
    """Classic t-shirt silhouette as an alpha mask."""
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)

    body = [
        (155, 210),  # left underarm join
        (110, 120),  # left shoulder tip
        (280, 55),   # left collar
        (400, 90),   # collar dip
        (520, 55),   # right collar
        (690, 120),  # right shoulder tip
        (645, 210),
        (620, 860),  # right hem
        (180, 860),  # left hem
    ]
    d.polygon(body, fill=255)

    # Sleeves
    d.polygon([(110, 120), (30, 300), (150, 360), (215, 220), (155, 210)], fill=255)
    d.polygon([(690, 120), (770, 300), (650, 360), (585, 220), (645, 210)], fill=255)

    # Collar cut-out
    d.ellipse([300, 30, 500, 130], fill=0)

    return mask.filter(ImageFilter.GaussianBlur(2))


def _hoodie_mask() -> Image.Image:
    """Boxier silhouette with a hood outline."""
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.polygon(
        [
            (140, 200), (90, 130), (270, 70), (400, 100), (530, 70),
            (710, 130), (660, 200), (640, 870), (160, 870),
        ],
        fill=255,
    )
    d.polygon([(90, 130), (15, 330), (140, 390), (205, 215), (140, 200)], fill=255)
    d.polygon([(710, 130), (785, 330), (660, 390), (595, 215), (660, 200)], fill=255)
    d.ellipse([250, 20, 550, 190], fill=255)   # hood volume
    d.ellipse([320, 60, 480, 160], fill=0)     # face opening
    return mask.filter(ImageFilter.GaussianBlur(2))


def _shade(base_rgb: tuple[int, int, int], mask: Image.Image) -> Image.Image:
    """Turn a flat silhouette into a shaded, fabric-like garment."""
    alpha = np.asarray(mask, dtype=np.float32) / 255.0

    # Vertical light falloff + a soft center highlight = cheap cloth shading.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vertical = 1.0 - 0.30 * (yy / H)
    highlight = 1.0 + 0.18 * np.exp(-(((xx - W / 2) / 260) ** 2 + ((yy - 260) / 300) ** 2))
    rng = np.random.default_rng(7)
    weave = 1.0 + rng.normal(0.0, 0.015, size=(H, W)).astype(np.float32)
    shading = np.clip(vertical * highlight * weave, 0.0, 1.35)

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        rgb[:, :, c] = np.clip(base_rgb[c] * shading, 0, 255)

    # Darken edges slightly so folds read at the silhouette boundary.
    edge = np.asarray(mask.filter(ImageFilter.GaussianBlur(10)), np.float32) / 255.0
    rgb *= (0.82 + 0.18 * edge)[:, :, None]

    out = np.dstack([rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


GARMENTS = {
    "tee_crimson": (_tshirt_mask, (196, 48, 58)),
    "tee_ocean": (_tshirt_mask, (36, 96, 176)),
    "tee_forest": (_tshirt_mask, (34, 120, 74)),
    "tee_charcoal": (_tshirt_mask, (56, 58, 64)),
    "hoodie_slate": (_hoodie_mask, (88, 96, 118)),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, (mask_fn, color) in GARMENTS.items():
        img = _shade(color, mask_fn())
        path = os.path.join(OUT_DIR, f"{name}.png")
        img.save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
