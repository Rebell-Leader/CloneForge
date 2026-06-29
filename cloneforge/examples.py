"""Examples library — reference objects with GROUND TRUTH for quality evaluation.

Two kinds of ground truth (per datasets research):
  - Standard parts (LEGO 2×4, ISO 7089 washer, DIN 934 nut, 16 mm die) have
    PUBLISHED EXACT DIMENSIONS → numeric "accurate to X mm" ground truth.
  - A reference mesh is also generated so we can report Chamfer / voxel-IoU.

We build the reference meshes parametrically from published specs and render a
clean preview that doubles as a demo input image (the live app also takes real
webcam/upload photos — photograph the real part and the published dims still apply).

`build_library(out_dir)` writes per-object reference.stl + preview.png + manifest.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh

from .render import render_single


# --- reference mesh builders (dims in mm, from published specs) -------------
def _lego_2x4():
    """LEGO brick 3001: body 31.8×15.8×9.6, studs Ø4.8 h1.8, pitch 8.0."""
    body = trimesh.creation.box(extents=[31.8, 15.8, 9.6])
    parts = [body]
    x0, y0 = -3 * 8.0 / 2, -8.0 / 2
    for ix in range(4):
        for iy in range(2):
            stud = trimesh.creation.cylinder(radius=2.4, height=1.8, sections=32)
            stud.apply_translation([x0 + ix * 8.0, y0 + iy * 8.0, 9.6 / 2 + 0.9])
            parts.append(stud)
    return trimesh.boolean.union(parts), "LEGO 2×4 brick (3001)"


def _washer_m10():
    """ISO 7089 M10: ID 10.5, OD 20, thickness 2.0."""
    outer = trimesh.creation.cylinder(radius=10.0, height=2.0, sections=64)
    inner = trimesh.creation.cylinder(radius=5.25, height=4.0, sections=64)
    return trimesh.boolean.difference([outer, inner]), "ISO 7089 flat washer (M10)"


def _hex_nut_m8():
    """DIN 934 M8: width across flats 13 (circumradius 7.51), height 6.5, bore 8."""
    body = trimesh.creation.cylinder(radius=7.506, height=6.5, sections=6)
    bore = trimesh.creation.cylinder(radius=4.0, height=10.0, sections=48)
    return trimesh.boolean.difference([body, bore]), "DIN 934 hex nut (M8)"


def _die_16():
    """16 mm six-sided die with shallow pips."""
    cube = trimesh.creation.box(extents=[16, 16, 16])
    return cube, "Six-sided die (16 mm)"


def _mug():
    outer = trimesh.creation.cylinder(radius=40, height=95, sections=64)
    inner = trimesh.creation.cylinder(radius=34, height=85, sections=64)
    inner.apply_translation([0, 0, 8])
    body = trimesh.boolean.difference([outer, inner])
    handle = trimesh.creation.torus(major_radius=22, minor_radius=6)
    handle.apply_translation([44, 0, 0])
    return trimesh.boolean.union([body, handle]), "Coffee mug"


def _spur_gear():
    """Complex case: 20-tooth spur gear — stresses primitive decomposition."""
    base = trimesh.creation.cylinder(radius=18, height=6, sections=64)
    parts = [base]
    for i in range(20):
        a = 2 * np.pi * i / 20
        tooth = trimesh.creation.box(extents=[5, 3, 6])
        tooth.apply_translation([20, 0, 0])
        tooth.apply_transform(trimesh.transformations.rotation_matrix(a, [0, 0, 1]))
        parts.append(tooth)
    bore = trimesh.creation.cylinder(radius=5, height=8, sections=48)
    return trimesh.boolean.difference([trimesh.boolean.union(parts), bore]), "20-tooth spur gear"


REGISTRY = [
    {"name": "washer_m10", "build": _washer_m10, "category": "simple",
     "goal": "Clone this washer as a 3D-printable model",
     "note": "ISO 7089 M10: OD 20, ID 10.5, thickness 2.0 mm"},
    {"name": "hex_nut_m8", "build": _hex_nut_m8, "category": "simple",
     "goal": "Clone this hex nut as a 3D-printable model",
     "note": "DIN 934 M8: across-flats 13, height 6.5 mm"},
    {"name": "die_16", "build": _die_16, "category": "simple",
     "goal": "Clone this die as a 3D-printable model",
     "note": "Standard 16 mm cube"},
    {"name": "lego_2x4", "build": _lego_2x4, "category": "medium",
     "goal": "Clone this LEGO brick as a 3D-printable model",
     "note": "LEGO 3001: 31.8×15.8×9.6 mm + 8 studs"},
    {"name": "mug", "build": _mug, "category": "medium",
     "goal": "Clone this mug as a 3D-printable model",
     "note": "~95 mm tall, 80 mm dia (approx)"},
    {"name": "spur_gear", "build": _spur_gear, "category": "complex",
     "goal": "Clone this gear as a 3D-printable model",
     "note": "20-tooth gear — shows the fidelity gap on complex geometry"},
]


def build_library(out_dir: str = "data/examples") -> dict:
    """Generate reference meshes + previews + manifest. Returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for spec in REGISTRY:
        d = os.path.join(out_dir, spec["name"])
        os.makedirs(d, exist_ok=True)
        mesh, title = spec["build"]()
        ref_stl = os.path.join(d, "reference.stl")
        preview = os.path.join(d, "preview.png")
        mesh.export(ref_stl)
        render_single(mesh, preview, view="iso")
        dims = sorted(round(float(x), 1) for x in (mesh.bounds[1] - mesh.bounds[0]))
        manifest.append({
            "name": spec["name"], "title": title, "category": spec["category"],
            "goal": spec["goal"], "note": spec["note"],
            "reference_stl": ref_stl, "preview": preview, "dims_mm": dims,
        })
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return {"out_dir": out_dir, "items": manifest}


def load_manifest(out_dir: str = "data/examples") -> list[dict]:
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        return build_library(out_dir)["items"]
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    lib = build_library()
    for it in lib["items"]:
        print(f"{it['category']:8} {it['name']:12} dims={it['dims_mm']} -> {it['preview']}")
