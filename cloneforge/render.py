"""Headless mesh rendering (pip-only, no display, no sudo).

matplotlib's Agg backend renders a Trimesh to PNG without OpenGL/X11 — the only
fully pip-installable, no-sudo path on this box. We add flat normal-based shading
(matplotlib 3D has no z-buffer, so plain polys look see-through) and a 4-view
composite (iso + front + side + top) that gives a VLM critic enough to judge shape.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

_VIEWS = {  # (elev, azim)
    "iso": (22, 45),
    "front": (0, -90),
    "side": (0, 0),
    "top": (90, -90),
}


def _load(mesh):
    if isinstance(mesh, str):
        m = trimesh.load(mesh, force="mesh")
    else:
        m = mesh
    return m


def _facecolors(m: trimesh.Trimesh, mode: str = "shaded", light=(0.4, 0.5, 0.75)) -> np.ndarray:
    """RGBA per-face colors.

    mode='shaded': flat shading from normal·light (CloneForge blue).
    mode='normal': normal-map colors (rgb = n*0.5+0.5) — exposes curvature/dents/non-flat
    faces that flat shading hides, helping the VLM critic judge geometry.
    """
    n = m.face_normals
    if mode == "normal":
        rgb = np.clip(n * 0.5 + 0.5, 0, 1)
    else:
        light = np.array(light) / np.linalg.norm(light)
        b = np.clip(np.abs(n @ light), 0.15, 1.0)
        rgb = np.clip(np.array([0.36, 0.56, 0.93])[None, :] * (0.35 + 0.65 * b[:, None]), 0, 1)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)


def _draw(ax, m, tris, colors, elev, azim):
    pc = Poly3DCollection(tris, facecolors=colors, edgecolors="none")
    ax.add_collection3d(pc)
    b = m.bounds
    ctr = (b[0] + b[1]) / 2
    r = (b[1] - b[0]).max() / 2 * 1.05
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def render_views(mesh, out_png: str, title: str | None = None, mode: str = "shaded") -> str:
    """Render a 2×2 composite (iso/front/side/top) of the mesh to out_png."""
    m = _load(mesh)
    tris = m.vertices[m.faces]
    colors = _facecolors(m, mode)
    fig = plt.figure(figsize=(6, 6))
    for i, (name, (elev, azim)) in enumerate(_VIEWS.items(), 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        _draw(ax, m, tris, colors, elev, azim)
        ax.set_title(name, fontsize=9, color="#444")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90, facecolor="white")
    plt.close(fig)
    return out_png


def render_single(mesh, out_png: str, view: str = "iso", mode: str = "shaded", angles=None) -> str:
    """Render one view to out_png. `angles=(elev, azim)` overrides the named view (for pose search)."""
    m = _load(mesh)
    tris = m.vertices[m.faces]
    colors = _facecolors(m, mode)
    elev, azim = angles if angles else _VIEWS.get(view, _VIEWS["iso"])
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, m, tris, colors, elev, azim)
    fig.savefig(out_png, dpi=90, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_png
