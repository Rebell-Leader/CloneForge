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
import urllib.request

import numpy as np
import trimesh

from .render import render_single

# Google Scanned Objects (CC-BY 4.0) via the kevinzakka/mujoco_scanned_objects mirror.
# Real-world 3D scans — used as ground-truth reference meshes (Chamfer/voxel-IoU) and to
# show the fidelity gap vs our parametric clones. Scans are in metres → scaled to mm.
GSO_BASE = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/main/models/{}/model.obj"
GSO = [
    {"name": "gso_mug", "gso": "ACE_Coffee_Mug_Kristen_16_oz_cup",
     "title": "Coffee mug (GSO scan)", "category": "real · medium",
     "goal": "Clone this mug as a 3D-printable model"},
    {"name": "gso_teapot", "gso": "Threshold_Porcelain_Teapot_White",
     "title": "Porcelain teapot (GSO scan)", "category": "real · complex",
     "goal": "Clone this teapot as a 3D-printable model"},
    {"name": "gso_panda", "gso": "Android_Figure_Panda",
     "title": "Android panda figure (GSO scan)", "category": "real · complex",
     "goal": "Clone this figurine as a 3D-printable model"},
]
GSO_ATTRIB = "Google Scanned Objects (CC-BY 4.0)"


def _load_gso(gso_name: str, dl_dir: str) -> trimesh.Trimesh:
    """Download a GSO model.obj (cached), scale m→mm, center, decimate if large."""
    os.makedirs(dl_dir, exist_ok=True)
    path = os.path.join(dl_dir, gso_name + ".obj")
    if not os.path.exists(path):
        urllib.request.urlretrieve(GSO_BASE.format(gso_name), path)
    m = trimesh.load(path, force="mesh")
    m.apply_scale(1000.0)
    m.apply_translation(-m.bounding_box.centroid)
    if len(m.faces) > 12000:
        try:
            m = m.simplify_quadric_decimation(face_count=12000)
        except Exception:  # noqa: BLE001
            pass
    return m


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


def _write_entry(out_dir, name, mesh, title, category, goal, note):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    ref_stl = os.path.join(d, "reference.stl")
    preview = os.path.join(d, "preview.png")
    mesh.export(ref_stl)
    render_single(mesh, preview, view="iso")
    dims = sorted(round(float(x), 1) for x in (mesh.bounds[1] - mesh.bounds[0]))
    return {"name": name, "title": title, "category": category, "goal": goal,
            "note": note, "reference_stl": ref_stl, "preview": preview, "dims_mm": dims}


def build_library(out_dir: str = "data/examples", include_gso: bool = True) -> dict:
    """Generate reference meshes + previews + manifest. GSO scans are best-effort
    (skipped if offline). Returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for spec in REGISTRY:
        mesh, title = spec["build"]()
        manifest.append(_write_entry(out_dir, spec["name"], mesh, title,
                                     spec["category"], spec["goal"], spec["note"]))
    if include_gso:
        dl = os.path.join(out_dir, "_gso_cache")
        for spec in GSO:
            try:
                mesh = _load_gso(spec["gso"], dl)
                manifest.append(_write_entry(out_dir, spec["name"], mesh, spec["title"],
                                             spec["category"], spec["goal"], GSO_ATTRIB))
            except Exception as e:  # noqa: BLE001 — offline / fetch failure is non-fatal
                print(f"  (skipped GSO {spec['name']}: {e})")
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
