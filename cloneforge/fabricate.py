"""Sandboxed execution of generator code -> validated STL/GLB mesh.

Generated Python runs in a restricted namespace exposing only trimesh + numpy.
On failure the stderr/exception is returned so the generator can self-repair.
"""
from __future__ import annotations

import importlib
import os
import re
import textwrap

import numpy as np
import trimesh


def _repair_hint(error: str) -> str:
    """Targeted guidance for known failure modes so the model doesn't repeat the mistake."""
    e = error.lower()
    tips = []
    if "syntaxerror" in e:
        tips.append("A comment likely wrapped onto a line without '#'. Remove comments entirely or "
                    "keep each on ONE line starting with '#'.")
    if "revolve" in e:
        tips.append("CORRECT revolve usage: `result = trimesh.creation.revolve(profile, sections=64)` "
                    "where profile = np.array([[r0,h0],[r1,h1],...]) of [radius,height] points. Pass "
                    "ONLY profile and sections=; do NOT pass angle/cap/any other positional arg.")
    if "apply_rotation" in e:
        tips.append("There is no apply_rotation; use "
                    "mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_rad,[x,y,z])).")
    if "polygon" in e or "extrude_polygon" in e:
        tips.append("extrude_polygon needs a shapely Polygon: "
                    "from shapely.geometry import Polygon; trimesh.creation.extrude_polygon(Polygon([(x,y),...]), height=H).")
    tips.append("If a richer builder keeps failing, FALL BACK to primitives "
                "(box/cylinder/sphere/torus) + trimesh.boolean.union/difference — those always work.")
    return " ".join(tips)


def _sanitize(code: str) -> str:
    """Strip markdown fences and normalize indentation from LLM-emitted code."""
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z0-9]*\n", "", code)
        code = re.sub(r"\n```$", "", code.rstrip())
    # common failure: whole block is uniformly indented -> dedent fixes it
    return _autofix_syntax(textwrap.dedent(code).strip() + "\n")


def _autofix_syntax(code: str, max_drops: int = 8) -> str:
    """Blank out lines that raise SyntaxError (e.g. a comment that wrapped without '#').
    Such lines are stray prose with no execution value; blanking keeps line numbers stable."""
    lines = code.split("\n")
    for _ in range(max_drops):
        try:
            compile("\n".join(lines), "<gen>", "exec")
            break
        except SyntaxError as exc:
            if exc.lineno and 1 <= exc.lineno <= len(lines) and lines[exc.lineno - 1].strip():
                lines[exc.lineno - 1] = ""
            else:
                break
    return "\n".join(lines)

_ALLOWED_IMPORTS = {"trimesh", "numpy", "math", "shapely"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__: only trimesh / numpy / math (and submodules)."""
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' is not allowed in the sandbox")
    return importlib.import_module(name)


def run_trimesh_code(code: str, out_dir: str, stem: str = "clone") -> tuple[bool, dict]:
    """Exec generator code, expecting it to assign `result` (a Trimesh).

    Returns (ok, info). On success info has stl_path, glb_path, and mesh stats.
    On failure info has {"error": <message>} for self-repair.
    """
    # Restricted globals: no builtins beyond a safe minimal set, only trimesh + np.
    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "abs": abs,
        "round": round, "float": float, "int": int, "list": list, "dict": dict,
        "tuple": tuple, "enumerate": enumerate, "zip": zip, "sum": sum,
        "__import__": _safe_import,
    }
    ns: dict = {"__builtins__": safe_builtins, "trimesh": trimesh, "np": np}
    code = _sanitize(code)
    try:
        exec(code, ns)  # noqa: S102 — sandboxed namespace, hackathon scope
    except Exception as e:  # noqa: BLE001
        return False, {"error": f"{type(e).__name__}: {e}"}

    result = ns.get("result")
    if not isinstance(result, trimesh.Trimesh):
        return False, {"error": "code did not assign a trimesh.Trimesh to `result`"}
    if result.is_empty or len(result.vertices) == 0:
        return False, {"error": "resulting mesh is empty"}

    os.makedirs(out_dir, exist_ok=True)
    stl_path = os.path.join(out_dir, f"{stem}.stl")
    glb_path = os.path.join(out_dir, f"{stem}.glb")
    result.export(stl_path)
    result.export(glb_path)

    bbox = (result.bounds[1] - result.bounds[0]).tolist()
    return True, {
        "stl_path": stl_path,
        "glb_path": glb_path,
        "stats": {
            "watertight": bool(result.is_watertight),
            "volume_mm3": round(float(result.volume), 1) if result.is_watertight else None,
            "bbox_mm": [round(b, 1) for b in bbox],
            "n_vertices": len(result.vertices),
            "n_faces": len(result.faces),
        },
    }


async def make_candidate(plan, spec, generator_fn, out_dir, stem, *, variant_hint=None, max_repairs=1):
    """Generate one candidate mesh (with light self-repair). Returns an info dict
    (stl_path/glb_path/stats/code/meta) or None. Used for best-of-N parallel generation."""
    feedback = variant_hint
    for _ in range(max_repairs + 1):
        artifact, meta = await generator_fn(plan, spec, feedback=feedback)
        ok, info = run_trimesh_code(artifact.code, out_dir, stem)
        if ok:
            info["code"], info["meta"] = artifact.code, meta
            return info
        feedback = (f"Your previous code failed: {info['error']}. {_repair_hint(info['error'])} "
                    "Return corrected complete code, no fences, no leading indentation, assign `result`.")
    return None


async def generate_mesh(plan, spec, generator_fn, out_dir: str, max_repairs: int = 3):
    """Generate code -> exec -> on failure feed error back to the generator (<=max_repairs).

    Yields (event_text, meta_or_none, info_or_none) tuples for streaming to the UI;
    the final yielded info dict (when ok) carries stl/glb paths + stats.
    """
    feedback = None
    last_code = ""
    for attempt in range(max_repairs + 1):
        artifact, meta = await generator_fn(plan, spec, feedback=feedback)
        last_code = artifact.code
        ok, info = run_trimesh_code(artifact.code, out_dir)
        if ok:
            info["code"] = last_code
            yield ("generator", meta, info)
            return
        # show the model its OWN broken code + the error so it can fix the exact line
        feedback = (f"Your previous code:\n```python\n{artifact.code}\n```\n"
                    f"failed with: {info['error']}\n{_repair_hint(info['error'])}\n"
                    "Return corrected COMPLETE code. No markdown fences, no leading indentation, "
                    "assign the final mesh to `result`.")
        yield (f"generator (repair {attempt + 1}: {info['error']})", meta, None)
    # exhausted repairs
    yield ("generator-failed", None, {"error": feedback, "code": last_code})
