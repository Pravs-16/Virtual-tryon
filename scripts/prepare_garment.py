"""Turn a photo of a real garment into a try-on ready PNG.

Usage (from the project root):
    python scripts/prepare_garment.py path/to/photo.jpg my_shirt

This crops the garment, removes the background, centers it on a
canvas sized for the overlay engine, and saves it into
static/garments/ so it appears on the rack after an app restart.

Best input photos: front-facing product shots on a plain background
("ghost mannequin" e-commerce photos are ideal). Automatic background
removal needs the `rembg` package (one-time install):

    pip install "rembg[cpu]"

If your image already has a transparent background (e.g. exported
from remove.bg or Photoshop), rembg isn't needed.
"""

import os
import sys

import numpy as np
from PIL import Image

CANVAS_W, CANVAS_H = 800, 900
MARGIN = 20
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static",
    "garments",
)


def _cutout(img: Image.Image) -> Image.Image:
    """Return an RGBA image with the background removed."""
    if img.mode == "RGBA":
        alpha = np.asarray(img)[:, :, 3]
        if alpha.min() < 250:  # already has real transparency
            return img
    try:
        from rembg import remove
    except ImportError:
        sys.exit(
            "This photo has no transparent background, so automatic\n"
            "background removal is needed. Install it once with:\n\n"
            '    pip install "rembg[cpu]"\n\n'
            "then re-run this command. (Or export a transparent PNG\n"
            "from remove.bg and use that as the input instead.)"
        )
    return remove(img.convert("RGB"))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src_path, name = sys.argv[1], sys.argv[2]
    if not os.path.isfile(src_path):
        sys.exit(f"File not found: {src_path}")

    img = _cutout(Image.open(src_path))

    # Tight-crop to the garment so image corners map to collar/hem.
    alpha = np.asarray(img)[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        sys.exit("Couldn't find a garment in this image (alpha is empty).")
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    img.thumbnail((CANVAS_W - 2 * MARGIN, CANVAS_H - 2 * MARGIN), Image.LANCZOS)
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    canvas.paste(
        img,
        ((CANVAS_W - img.width) // 2, (CANVAS_H - img.height) // 2),
        img,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.png")
    canvas.save(out_path)
    print(f"wrote {out_path}")
    print("Restart app.py and it will appear on the rack.")


if __name__ == "__main__":
    main()
