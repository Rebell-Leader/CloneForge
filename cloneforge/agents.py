"""The four specialized agents. Each = system prompt + strict-schema call.

No tool calling anywhere (it can't combine with image inputs on Gemma 4) — every
agent exchanges validated JSON via structured outputs.
"""
from __future__ import annotations

import json

from . import llm
from .schemas import (
    BestSelection,
    CritiqueVerdict,
    FabPlan,
    GeneratedArtifact,
    VisionSpec,
    response_format,
)

VISION_SYS = (
    "You are a precision vision-analysis agent for digital fabrication. Examine the "
    "image and produce a rigorous physical specification of the PRIMARY object. "
    "Estimate real-world dimensions in millimetres using visible scale cues. Note "
    "materials, salient features, and any defects. Be concrete and quantitative."
)

PLANNER_SYS = (
    "You are a fabrication planner. Given an object spec, produce a constructive plan that "
    "reproduces the object. Choose the BEST construction strategy and say which in `fab_method`/"
    "`notes`:\n"
    "• Rotationally-symmetric objects (mug, bottle, vase, bowl, cup, lamp, wheel) → a SURFACE OF "
    "REVOLUTION: describe the radius/height profile.\n"
    "• Prismatic objects with a constant cross-section (gear, bracket, sign, key) → an EXTRUDED 2D "
    "profile: describe the cross-section outline.\n"
    "• Otherwise → a composition of PARAMETRIC PRIMITIVES (box, cylinder, sphere, torus) with "
    "boolean add/subtract.\n"
    "Keep it simple and physically buildable. Dimensions in millimetres. box dims=[x,y,z]; "
    "cylinder=[radius,height]; sphere=[radius]; torus=[major_R,minor_r]. The `primitives` list is "
    "for the primitive strategy; for revolve/extrude, describe the profile in `steps`/`notes`."
)

GENERATOR_SYS = (
    "You are a 3D code generator. Output Python that builds the planned object with "
    "`trimesh` and `np` (numpy), assigning the final mesh to a variable named `result`. "
    "Use trimesh.creation.box(extents=[x,y,z]) / cylinder(radius,height,sections=64) / "
    "icosphere(radius=r) / torus(major_radius=R, minor_radius=r), .apply_translation([x,y,z]), "
    "and trimesh.boolean.union/difference([...]). Units are millimetres. "
    "To ROTATE a mesh use mesh.apply_transform(trimesh.transformations.rotation_matrix(angle_rad, "
    "[x,y,z])) — there is NO apply_rotation method.\n"
    "RICHER BUILDERS for non-primitive shapes (prefer these when they fit):\n"
    "• Surface of revolution (mug/bottle/vase/bowl/cup/lamp): build a profile of [radius, height] "
    "points and revolve it — profile=np.array([[r0,h0],[r1,h1],...]); result=trimesh.creation."
    "revolve(profile, sections=64). The argument is a list of [radius,height] points (radius first); "
    "omit `angle` for a full 360° solid; close the profile (radius 0 at top/bottom) for watertight.\n"
    "• Extruded 2D cross-section (gear/bracket/sign/key): from shapely.geometry import Polygon; "
    "result=trimesh.creation.extrude_polygon(Polygon([(x,y),...]), height=H).\n"
    "• Also available: trimesh.creation.cone(radius,height), capsule(height,radius), "
    "annulus(r_min,r_max,height), sweep_polygon(Polygon([...]), path_points).\n"
    "Allowed imports: trimesh, numpy as np, math, shapely. No file I/O, no printing — only build "
    "`result`. Keep it watertight.\n"
    "CODE STYLE: write minimal code. Use FEW comments and keep every comment on ONE line starting "
    "with '#'. NEVER wrap a comment across two lines (a continuation line without '#' is a syntax error)."
)

CRITIC_SYS = (
    "You are a fabrication critic. Compare the produced mesh statistics against the "
    "object spec and plan. Judge geometric fidelity and printability. Approve only if "
    "the mesh is watertight and reasonably matches the object; otherwise give concrete "
    "fix instructions for the generator."
)

VISUAL_CRITIC_SYS = (
    "You are a visual fabrication critic with eyes. You are shown the ORIGINAL object "
    "photo(s) and a multi-view RENDER of the candidate 3D model the system generated. "
    "Compare them directly. Do NOT give a vibe check — produce a concrete, checkable diff: "
    "for each discrepancy in overall shape, proportions, COUNT of features (holes, handles, "
    "legs, ribs), presence/absence of parts, and relative sizes, state what is wrong and "
    "which generator change fixes it (e.g. 'handle too thick — reduce torus minor_radius', "
    "'missing the spout', 'body should be ~30% taller'). Approve only when the render's "
    "silhouette and feature set clearly match the photo."
)


def _parse(text: str, model):
    return model.model_validate_json(text)


async def vision_agent(image_data_uris, goal: str, view_labels: list[str] | None = None):
    """Analyze 1..5 images of the SAME object. Multiple views (front/side/top) sharply
    improve depth/proportion estimates — use the side view for depth, top for footprint."""
    if isinstance(image_data_uris, str):
        image_data_uris = [image_data_uris]
    multi = len(image_data_uris) > 1
    prompt = f"Analyze this object for the goal: {goal}."
    if multi:
        prompt += (" You are given multiple views of the SAME object. Cross-reference them: "
                   "use the side view to judge depth/thickness and the top view for the footprint. "
                   "Reconcile the views into one consistent specification.")
    msgs = [
        {"role": "system", "content": VISION_SYS},
        {"role": "user", "content": llm.multi_image_content(prompt, image_data_uris, view_labels)},
    ]
    text, meta = await llm.acall(msgs, schema=response_format(VisionSpec), max_tokens=1500)
    return _parse(text, VisionSpec), meta


async def visual_critic_agent(spec, original_uris, render_uris, mesh_stats: dict):
    """Multimodal critic: SEE the original photo(s) + renders of the candidate mesh (shaded +
    normal-map) and produce a concrete diff. The main fidelity lever (render→VLM→fix loop).
    The normal-map view exposes curvature/flatness errors flat shading hides."""
    if isinstance(original_uris, str):
        original_uris = [original_uris]
    if isinstance(render_uris, str):
        render_uris = [render_uris]
    render_labels = ["YOUR MODEL render — shaded (4 views)", "YOUR MODEL render — normal map (4 views)"]
    uris = list(original_uris) + list(render_uris)
    labels = ([f"ORIGINAL photo {i + 1}" for i in range(len(original_uris))]
              + render_labels[:len(render_uris)])
    prompt = (
        f"Target object: {spec.object}. Mesh stats: {json.dumps(mesh_stats)}.\n"
        "Compare the ORIGINAL photo(s) to YOUR model renders. For each discrepancy, name the "
        "generator change that fixes it in the form 'problem → primitive/param + direction' "
        "(e.g. 'handle too thick → reduce torus minor_radius'). Cover shape, proportions, "
        "feature counts, missing/extra parts, and relative sizes."
    )
    msgs = [
        {"role": "system", "content": VISUAL_CRITIC_SYS},
        {"role": "user", "content": llm.multi_image_content(prompt, uris, labels)},
    ]
    # NOTE: structured output + images is fine; tool-calling + images is NOT (Gemma 4).
    text, meta = await llm.acall(msgs, schema=response_format(CritiqueVerdict), max_tokens=1500)
    return _parse(text, CritiqueVerdict), meta


async def planner_agent(spec: VisionSpec, goal: str):
    msgs = [
        {"role": "system", "content": PLANNER_SYS},
        {"role": "user", "content":
            f"Goal: {goal}\n\nObject spec:\n{spec.model_dump_json(indent=2)}"},
    ]
    # NOTE: reasoning_effort is intentionally OFF here. For Gemma 4 the levels are
    # equivalent, and enabling it destabilizes structured (json_schema) output —
    # reasoning tokens can crowd out the JSON and return empty content.
    text, meta = await llm.acall(
        msgs, schema=response_format(FabPlan), max_tokens=3000)
    return _parse(text, FabPlan), meta


async def generator_agent(plan: FabPlan, spec: VisionSpec, feedback: str | None = None):
    user = f"Object: {spec.object}\n\nPlan:\n{plan.model_dump_json(indent=2)}"
    if feedback:
        user += f"\n\nThe previous attempt failed. Fix it:\n{feedback}"
    msgs = [
        {"role": "system", "content": GENERATOR_SYS},
        {"role": "user", "content": user},
    ]
    text, meta = await llm.acall(
        msgs, schema=response_format(GeneratedArtifact), max_tokens=3000)
    return _parse(text, GeneratedArtifact), meta


SELECTOR_SYS = (
    "You select the best 3D-model candidate. You see the ORIGINAL object photo and several "
    "candidate renders. Pick the 0-based index whose shape, proportions, and feature set best "
    "match the photo."
)


async def select_best_agent(spec, original_uris, candidate_render_uris, hints=None):
    """One multimodal call: pick the best candidate render index vs the photo (best-of-N).
    `hints` (optional) adds a numeric note per candidate, e.g. silhouette-IoU scores."""
    if isinstance(original_uris, str):
        original_uris = [original_uris]
    uris = list(original_uris) + list(candidate_render_uris)
    labels = (["ORIGINAL photo"] * len(original_uris)
              + [f"Candidate {i}" + (f" — {hints[i]}" if hints and i < len(hints) else "")
                 for i in range(len(candidate_render_uris))])
    prompt = (f"Target object: {spec.object}. Choose the 0-based index of the candidate that best "
              "matches the photo in shape, proportions, and features.")
    msgs = [
        {"role": "system", "content": SELECTOR_SYS},
        {"role": "user", "content": llm.multi_image_content(prompt, uris, labels)},
    ]
    text, meta = await llm.acall(msgs, schema=response_format(BestSelection), max_tokens=500)
    return _parse(text, BestSelection), meta


async def critic_agent(spec: VisionSpec, plan: FabPlan, mesh_stats: dict):
    msgs = [
        {"role": "system", "content": CRITIC_SYS},
        {"role": "user", "content":
            f"Object spec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Plan notes: {plan.notes}\n\n"
            f"Produced mesh stats:\n{json.dumps(mesh_stats, indent=2)}"},
    ]
    text, meta = await llm.acall(msgs, schema=response_format(CritiqueVerdict), max_tokens=1200)
    return _parse(text, CritiqueVerdict), meta
