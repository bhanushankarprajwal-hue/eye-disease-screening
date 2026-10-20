"""
Simple, fast image-quality check for fundus photos (numpy + Pillow only, no model).

It answers one question before the AI looks at a photo: "is this photo usable?"
Only clear technical failures are marked "retake" (too dark, over-exposed, almost no fundus visible,
probably not a fundus photo -- wrong colour, or no round field of view with dark corners). Softer problems (blur / low detail, uneven lighting) are only shown as
warnings, because some eye diseases -- cataract above all -- make a photo look hazy by themselves, and a
quality check must never hide a diseased eye from the doctor.

Thresholds were calibrated on the ODIR-5K raw photos (Data/quality_thresholds.json, written by
Code/calibrate_quality.py); defaults below are used if that file is missing.
"""
import json
import os

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS_FILE = os.path.join(PROJECT_ROOT, "Data", "quality_thresholds.json")

DEFAULTS = {
    "min_fundus_area": 0.15,      # fraction of the photo that is not black background
    "min_brightness": 35.0,       # mean brightness of the fundus area (0-255)
    "max_saturated": 0.20,        # fraction of fundus pixels that are blown-out white
    "min_detail": 2.0,            # fine-detail score (vessel/edge strength) -> warning only
    "max_uneven": 0.55,           # brightness difference between fundus halves -> warning only
    "min_redness": 1.02,          # fundus photos are red/orange (R > G) -> below = probably not a fundus
    "max_corner": 60.0,           # fundus photos have a round field with dark corners -> bright corners = not a fundus
}


def load_thresholds():
    t = dict(DEFAULTS)
    if os.path.isfile(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE) as f:
            t.update(json.load(f).get("thresholds", {}))
    return t


def _small_rgb(img_or_path, size=256):
    if isinstance(img_or_path, Image.Image):
        img = img_or_path.copy()
    else:
        img = Image.open(img_or_path)
        if img.format == "JPEG":
            img.draft("RGB", (size * 2, size * 2))   # fast JPEG decode at reduced size
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.BILINEAR)
    return img


def measure(img_or_path):
    """Returns the raw quality measurements for one photo (all cheap: ~5-20 ms per photo)."""
    img = _small_rgb(img_or_path)
    a = np.asarray(img).astype(np.float32)
    if a[..., 2].mean() > a[..., 0].mean() + 10:   # colour channels saved in BGR order -> swap back
        a = a[..., ::-1]                            # (same rule as predict_multiclass.load_rgb)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mask = lum > 20
    area = float(mask.mean())
    k = max(4, int(0.08 * min(lum.shape)))
    corner = float(np.mean([lum[:k, :k].mean(), lum[:k, -k:].mean(), lum[-k:, :k].mean(), lum[-k:, -k:].mean()]))
    if mask.sum() < 50:
        return {"fundus_area": area, "brightness": 0.0, "saturated": 0.0, "detail": 0.0,
                "uneven": 0.0, "redness": 0.0, "corner": corner}
    brightness = float(lum[mask].mean())
    saturated = float((lum[mask] > 235).mean())
    # fine detail: green channel minus a blurred copy (vessels / edges), measured inside the fundus,
    # with the rim eroded away so the black border edge is not counted as detail
    gimg = Image.fromarray(g.astype(np.uint8))
    hp = np.abs(g - np.asarray(gimg.filter(ImageFilter.GaussianBlur(3))).astype(np.float32))
    inner = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(9))) > 0
    detail = float(hp[inner].mean()) if inner.sum() > 50 else 0.0
    # uneven lighting: difference between the brightest and darkest half of the fundus
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    halves = [lum[mask & (np.arange(lum.shape[0])[:, None] < cy)], lum[mask & (np.arange(lum.shape[0])[:, None] >= cy)],
              lum[mask & (np.arange(lum.shape[1])[None, :] < cx)], lum[mask & (np.arange(lum.shape[1])[None, :] >= cx)]]
    means = [h.mean() for h in halves if h.size]
    uneven = float((max(means) - min(means)) / max(brightness, 1.0))
    redness = float(r[mask].mean() / max(g[mask].mean(), 1.0))
    return {"fundus_area": area, "brightness": brightness, "saturated": saturated, "detail": detail,
            "uneven": uneven, "redness": redness, "corner": corner}


def assess(img_or_path, thresholds=None):
    """Returns (status, problems, measurements). status is 'ok', 'warning' or 'retake'."""
    t = thresholds or load_thresholds()
    m = measure(img_or_path)
    retake, warn = [], []
    if m["fundus_area"] < t["min_fundus_area"]:
        retake.append("too dark / almost no fundus visible")
    else:
        if m["brightness"] < t["min_brightness"]:
            retake.append("too dark")
        if m["saturated"] > t["max_saturated"]:
            retake.append("over-exposed (too bright)")
        if m["redness"] < t["min_redness"] or m["corner"] > t["max_corner"]:
            retake.append("may not be a fundus photo")
        if m["detail"] < t["min_detail"]:
            warn.append("blurry / low detail")
        if m["uneven"] > t["max_uneven"]:
            warn.append("uneven lighting")
    status = "retake" if retake else ("warning" if warn else "ok")
    return status, retake + warn, m
