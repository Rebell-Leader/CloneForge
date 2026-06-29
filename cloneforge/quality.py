"""Quantitative mesh quality / similarity metrics (all verified on this venv).

Two modes:
  - vs known target dimensions (always available): OBB-based dimension match.
  - vs a reference mesh (GSO etc.): Chamfer distance + volumetric IoU after
    normalization and point-to-point ICP alignment.

Pip-only (trimesh + numpy + scipy). No GL, no sudo, no rtree.
Key correctness notes (verified):
  - Use the ORIENTED bounding box (OBB) for dimensions; the axis-aligned box
    inflates under rotation.
  - ICP must be point-to-point (mesh-based ICP needs rtree, which isn't installed).
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _load(mesh) -> trimesh.Trimesh:
    return trimesh.load(mesh, force="mesh") if isinstance(mesh, str) else mesh


def _obb_extents(m: trimesh.Trimesh) -> np.ndarray:
    """Orientation-invariant real-world dimensions (sorted ascending)."""
    return np.sort(m.bounding_box_oriented.primitive.extents)


# --------------------------------------------------------------------------
# vs target dimensions (mm) — cheap, always available
# --------------------------------------------------------------------------
def dimension_score(mesh, target_dims_mm: list[float]) -> dict:
    """Compare mesh OBB to a target [x,y,z] in mm (order-insensitive)."""
    m = _load(mesh)
    got = _obb_extents(m)
    tgt = np.sort(np.array(target_dims_mm, dtype=float))
    tgt[tgt == 0] = 1e-6
    rel_err = np.abs(got - tgt) / tgt
    score = float(np.clip(1 - rel_err.mean(), 0, 1))
    return {
        "dims_got_mm": [round(float(x), 1) for x in got],
        "dims_target_mm": [round(float(x), 1) for x in tgt],
        "axis_ratio": [round(float(x), 2) for x in (got / tgt)],
        "dimension_score": round(score, 3),
    }


# --------------------------------------------------------------------------
# vs reference mesh — Chamfer + voxel IoU after normalize + ICP
# --------------------------------------------------------------------------
def _normalize(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = m.copy()
    m.apply_translation(-m.bounding_box.centroid)
    s = m.extents.max()
    if s > 0:
        m.apply_scale(1.0 / s)
    return m


def _icp_align(target: trimesh.Trimesh, moving: trimesh.Trimesh, n: int = 4000) -> trimesh.Trimesh:
    """Point-to-point ICP (no rtree): align `moving` onto `target`."""
    T, _, _ = trimesh.registration.icp(moving.sample(n), target.sample(n), max_iterations=50)
    out = moving.copy()
    out.apply_transform(T)
    return out


def chamfer_distance(a, b, n: int = 5000) -> float:
    """Symmetric mean Chamfer distance (normalized scale). Lower = better."""
    pa, pb = a.sample(n), b.sample(n)
    d_ab = cKDTree(pb).query(pa)[0]
    d_ba = cKDTree(pa).query(pb)[0]
    return float(d_ab.mean() + d_ba.mean())


def voxel_iou(a, b, pitch: float = 0.02) -> float:
    """Volumetric IoU on filled voxel grids in a shared frame. Higher = better."""
    pa = a.voxelized(pitch).fill().points
    pb = b.voxelized(pitch).fill().points
    sa = {tuple(p) for p in np.round(pa / pitch).astype(int)}
    sb = {tuple(p) for p in np.round(pb / pitch).astype(int)}
    u = len(sa | sb)
    return float(len(sa & sb) / u) if u else 0.0


def compare(mesh, reference=None, target_dims_mm=None) -> dict:
    """Run whichever metrics are applicable; return a compact report."""
    out: dict = {}
    if target_dims_mm is not None:
        out.update(dimension_score(mesh, target_dims_mm))
    if reference is not None:
        try:
            ref = _normalize(_load(reference))
            gen = _icp_align(ref, _normalize(_load(mesh)))
            out["chamfer"] = round(chamfer_distance(ref, gen), 4)
            out["voxel_iou"] = round(voxel_iou(ref, gen), 3)
        except Exception as e:  # noqa: BLE001
            out["reference_error"] = str(e)
    return out
