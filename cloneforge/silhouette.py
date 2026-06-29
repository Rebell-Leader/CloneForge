"""Silhouette-IoU numeric gate (pip-free).

A cheap, model-free geometric signal: render the candidate mesh from canonical views,
extract each silhouette, and compare (best-IoU) to the input object's silhouette. Used to
rank best-of-N candidates numerically and to report a "shape match vs photo" score.

Foreground extraction is background-color thresholding (robust for clean/white-bg photos
and our renders). For cluttered photos, `pip install rembg` and swap in a matting model —
the API here (a boolean mask) stays the same. Masks are normalized to their bbox before
IoU, so the score is scale/translation-invariant (not rotation — we take the best over views).
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

from .render import render_single

_VIEWS = ("iso", "front", "side", "top")


def _img_from_uri(uri: str) -> Image.Image:
    b64 = uri.split(",", 1)[1] if "," in uri else uri
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _foreground_mask(im: Image.Image) -> np.ndarray:
    """Boolean object mask via background-color (corner) thresholding."""
    a = np.asarray(im.convert("RGB")).astype(float)
    h, w, _ = a.shape
    c = 8
    corners = np.concatenate([a[:c, :c].reshape(-1, 3), a[:c, -c:].reshape(-1, 3),
                              a[-c:, :c].reshape(-1, 3), a[-c:, -c:].reshape(-1, 3)])
    bg = corners.mean(0)
    dist = np.linalg.norm(a - bg, axis=2)
    t = max(28.0, float(dist.mean()))
    return dist > t


def _normalize(mask: np.ndarray, size: int = 128) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((size, size), bool)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((crop * 255).astype("uint8")).resize((size, size))
    return np.asarray(im) > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 0.0


def _silhouette_at(stl: str, out_dir: str, view=None, angles=None) -> np.ndarray:
    png = render_single(stl, os.path.join(out_dir, "_sil.png"), view=view or "iso", angles=angles)
    a = np.asarray(Image.open(png).convert("RGB"))
    return a.min(axis=2) < 240  # non-white = object


def _load_target(target_image) -> np.ndarray:
    if isinstance(target_image, Image.Image):
        im = target_image
    elif isinstance(target_image, str) and target_image.startswith("data:"):
        im = _img_from_uri(target_image)
    else:
        im = Image.open(target_image)
    return _normalize(_foreground_mask(im))


def silhouette_iou(stl: str, target_image, out_dir: str = "outputs") -> float:
    """Best silhouette IoU over the 4 canonical views (fast; for best-of-N ranking). 0..1."""
    os.makedirs(out_dir, exist_ok=True)
    tgt = _load_target(target_image)
    best = 0.0
    for v in _VIEWS:
        try:
            best = max(best, iou(tgt, _normalize(_silhouette_at(stl, out_dir, view=v))))
        except Exception:  # noqa: BLE001
            continue
    return round(best, 3)


# Lightweight camera-pose estimation: search a viewpoint grid for the angle whose rendered
# silhouette best matches the input (the discrete render-and-compare pose search from One-2-3-45).
_POSE_GRID = [(e, a) for e in (-15, 15, 45) for a in range(0, 360, 45)]  # 3×8 = 24


def estimate_pose(stl: str, target_image, out_dir: str = "outputs"):
    """Return (best_iou, (elev, azim)) — the viewpoint whose silhouette best matches the photo."""
    os.makedirs(out_dir, exist_ok=True)
    tgt = _load_target(target_image)
    best = (0.0, (15, 45))
    for elev, azim in _POSE_GRID:
        try:
            sc = iou(tgt, _normalize(_silhouette_at(stl, out_dir, angles=(elev, azim))))
        except Exception:  # noqa: BLE001
            continue
        if sc > best[0]:
            best = (round(sc, 3), (elev, azim))
    return best
