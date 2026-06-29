"""Plain async orchestrator: vision -> plan -> generate(+repair) -> VISUAL critic loop.

Async generator that yields incremental state so Gradio can stream the agent
transcript live. Key design points:
  - Vision is computed ONCE and cached across critic iterations (30 rpm budget).
  - The critic is VISUAL: it renders the candidate mesh and compares it to the
    original photo(s) — the main fidelity lever (render -> VLM -> fix loop).
  - Multi-image input (front/side/top) is supported and improves the vision spec.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from . import agents, quality, silhouette
from .fabricate import generate_mesh, make_candidate
from .llm import encode_image
from .render import render_single, render_views


@dataclass
class CloneState:
    transcript: list[dict] = field(default_factory=list)  # gr.Chatbot messages
    spec: Any = None   # cached VisionSpec (reused by refine — saves a vision call)
    plan: Any = None   # cached FabPlan
    image_uris: list = field(default_factory=list)  # originals (for refine's visual critic)
    glb_path: str | None = None
    stl_path: str | None = None
    render_png: str | None = None
    code: str | None = None
    stats: dict | None = None
    quality: dict | None = None
    total_calls: int = 0
    total_latency_s: float = 0.0
    done: bool = False
    error: str | None = None


def _render_pair(stl_path: str, out_dir: str):
    """Render shaded + normal-map composites; return (shaded_png, [shaded_uri, normal_uri])."""
    shaded = render_views(stl_path, os.path.join(out_dir, "render.png"), mode="shaded")
    normal = render_views(stl_path, os.path.join(out_dir, "render_normal.png"), mode="normal")
    return shaded, [encode_image(shaded), encode_image(normal)]


def _msg(state: CloneState, agent: str, content: str, meta=None):
    if meta is not None:
        state.total_calls += 1
        state.total_latency_s += meta.latency_s
        tag = f"{meta.provider} {meta.latency_s:.2f}s"
        if meta.fell_back:
            tag += " ↪fallback"
        if meta.extra.get("retries"):
            tag += f" (retried {meta.extra['retries']}×)"
        title = f"{agent} · {tag}"
    else:
        title = agent
    state.transcript.append(
        {"role": "assistant", "content": content, "metadata": {"title": title}}
    )


async def _build(state, plan, spec, gen_fn, out_dir, label):
    """Run generate_mesh, update state + transcript. Returns True if a mesh was produced."""
    ok = False
    async for txt, gmeta, info in generate_mesh(plan, spec, gen_fn, out_dir):
        if info and "stl_path" in info:
            state.glb_path, state.stl_path = info["glb_path"], info["stl_path"]
            state.code, state.stats = info["code"], info["stats"]
            _msg(state, label, f"Mesh built ✓\n```\n{_fmt_stats(info['stats'])}\n```", gmeta)
            ok = True
        elif info and "error" in info:
            state.error = info["error"]
            _msg(state, label, f"Failed after repairs: {info['error']}", gmeta)
        else:
            _msg(state, label, txt, gmeta)
        yield state
    state._last_build_ok = ok  # type: ignore[attr-defined]


async def clone_pipeline(
    image_data_uris,
    goal: str,
    *,
    max_iters: int = 2,
    out_dir: str = "outputs",
    target_dims_mm=None,
    reference_mesh=None,
    n_candidates: int = 1,
):
    """Async generator yielding CloneState snapshots as each agent acts."""
    if isinstance(image_data_uris, str):
        image_data_uris = [image_data_uris]
    os.makedirs(out_dir, exist_ok=True)
    state = CloneState(image_uris=list(image_data_uris))

    # 1) Vision (computed once, cached) -----------------------------------
    spec, meta = await agents.vision_agent(image_data_uris, goal)
    state.spec = spec
    _msg(state, "👁 Vision", _fmt_spec(spec), meta)
    yield state

    # 2) Planner ----------------------------------------------------------
    plan, meta = await agents.planner_agent(spec, goal)
    state.plan = plan
    _msg(state, "🧠 Planner", _fmt_plan(plan), meta)
    yield state

    # 3) Generate -------------------------------------------------------
    if n_candidates > 1:
        async for s in _best_of_n(state, plan, spec, image_data_uris, n_candidates, out_dir):
            yield s
    else:
        async for s in _build(state, plan, spec, agents.generator_agent, out_dir, "🛠 Generator"):
            yield s
    if state.stl_path is None:
        state.done = True
        yield state
        return

    # 4) Visual critic loop (render -> compare to photo -> fix) -----------
    for i in range(max_iters):
        state.render_png, render_uris = _render_pair(state.stl_path, out_dir)
        verdict, meta = await agents.visual_critic_agent(spec, image_data_uris, render_uris, state.stats)
        mark = "✅ approved" if verdict.approved else "🔁 revise"
        issues = "\n".join(f"• {x}" for x in verdict.issues[:5])
        _msg(state, f"🔎 Visual Critic #{i + 1}",
             f"{mark} (score {verdict.score:.2f})\n{issues}\n→ {verdict.fix_instructions}", meta)
        yield state
        if verdict.approved:
            break
        gen = _with_feedback(verdict.fix_instructions, agents.generator_agent)
        async for s in _build(state, plan, spec, gen, out_dir, "🛠 Generator (revised)"):
            yield s

    # final render + quality metrics -------------------------------------
    if state.stl_path:
        state.render_png = render_views(state.stl_path, os.path.join(out_dir, "render.png"))
        q = {}
        if target_dims_mm is not None or reference_mesh is not None:
            q.update(quality.compare(state.stl_path, reference=reference_mesh, target_dims_mm=target_dims_mm))
        if image_data_uris:
            try:
                iou_score, (elev, azim) = silhouette.estimate_pose(state.stl_path, image_data_uris[0], out_dir)
                q["silhouette_iou"] = iou_score
                q["viewpoint"] = f"elev {elev}° · azim {azim}°"
            except Exception:  # noqa: BLE001
                pass
        if q:
            state.quality = q
            _msg(state, "📊 Quality", _fmt_quality(q))
    state.done = True
    yield state


_VARIANTS = [
    None,
    "Variant: prefer fewer, larger primitives for a cleaner solid.",
    "Variant: capture finer features with extra small primitives.",
    "Variant: emphasize correct overall proportions over detail.",
]


async def _best_of_n(state, plan, spec, image_uris, n, out_dir):
    """Generate n candidates in parallel, render each, let the critic pick the best (one call)."""
    n = min(n, len(_VARIANTS), 4)  # ≤4 so photo+candidates ≤5 images
    _msg(state, "🛠 Generator", f"Generating {n} candidates in parallel…")
    yield state
    infos = await asyncio.gather(*[
        make_candidate(plan, spec, agents.generator_agent, out_dir, f"cand{i}", variant_hint=_VARIANTS[i])
        for i in range(n)
    ])
    cands = [c for c in infos if c]
    if not cands:
        state.error = "all candidates failed"
        _msg(state, "🛠 Generator", "All candidates failed to build.")
        yield state
        return
    # render each candidate (shaded) + a numeric silhouette-IoU gate vs the photo
    render_uris, hints = [], []
    photo = image_uris[0] if image_uris else None
    for i, c in enumerate(cands):
        png = render_single(c["stl_path"], os.path.join(out_dir, f"cand{i}.png"))
        c["preview"] = png
        render_uris.append(encode_image(png))
        c["sil"] = silhouette.silhouette_iou(c["stl_path"], photo, out_dir) if photo else 0.0
        hints.append(f"silhouette match {c['sil']:.0%}")
    if len(cands) == 1:
        best, meta, idx = cands[0], None, 0
    else:
        sel, meta = await agents.select_best_agent(spec, image_uris, render_uris, hints=hints)
        idx = sel.best_index if 0 <= sel.best_index < len(cands) else max(
            range(len(cands)), key=lambda j: cands[j]["sil"])  # fallback: best silhouette
        best = cands[idx]
    state.glb_path, state.stl_path = best["glb_path"], best["stl_path"]
    state.code, state.stats = best["code"], best["stats"]
    extra = (f"chose #{idx}/{len(cands)} (silhouette {best['sil']:.0%}): {sel.reason}"
             if len(cands) > 1 else "")
    _msg(state, "🏅 Selector", f"Built {len(cands)}/{n} candidates. {extra}\n```\n{_fmt_stats(best['stats'])}\n```", meta)
    yield state


async def refine_pipeline(
    state: CloneState,
    user_instruction: str,
    *,
    out_dir: str = "outputs",
    target_dims_mm=None,
    reference_mesh=None,
    extra_uris=None,
):
    """User-driven correction: reuse the cached spec+plan and regenerate with the
    user's text instruction as feedback, then one visual-critic + quality pass.
    Skips vision+planner (saves rpm budget) — the 'iterative design' use case."""
    if state.spec is None or state.plan is None:
        _msg(state, "⚠️ Refine", "Run a clone first, then refine it.")
        yield state
        return
    state.done = False
    _msg(state, "🙋 Your correction", user_instruction)
    yield state

    gen = _with_feedback(f"User correction (apply precisely): {user_instruction}",
                         agents.generator_agent)
    async for s in _build(state, state.plan, state.spec, gen, out_dir, "🛠 Generator (refine)"):
        yield s
    if state.stl_path is None:
        state.done = True
        yield state
        return

    state.render_png, render_uris = _render_pair(state.stl_path, out_dir)
    originals = list(extra_uris or []) + (state.image_uris or [render_uris[0]])
    verdict, meta = await agents.visual_critic_agent(
        state.spec, originals, render_uris, state.stats)
    mark = "✅ matches request" if verdict.approved else "↩ still off"
    _msg(state, "🔎 Visual Critic", f"{mark} (score {verdict.score:.2f})\n{verdict.fix_instructions}", meta)
    yield state

    if target_dims_mm is not None or reference_mesh is not None:
        state.quality = quality.compare(state.stl_path, reference=reference_mesh, target_dims_mm=target_dims_mm)
        _msg(state, "📊 Quality", _fmt_quality(state.quality))
    state.done = True
    yield state


def _with_feedback(critic_feedback: str, gen_fn):
    async def wrapped(plan, spec, feedback=None):
        combined = critic_feedback if not feedback else f"{critic_feedback}\nAlso: {feedback}"
        return await gen_fn(plan, spec, feedback=combined)
    return wrapped


# --- formatting helpers ----------------------------------------------------
def _fmt_spec(s) -> str:
    d = s.dimensions
    return (f"**{s.object}** ({s.confidence:.0%})\n{s.geometry}\n"
            f"~{d.height_mm:.0f}×{d.width_mm:.0f}×{d.depth_mm:.0f} mm · {', '.join(s.materials)}\n"
            f"features: {', '.join(s.features) or '—'}"
            + (f"\ndefects: {', '.join(s.defects)}" if s.defects else ""))


def _fmt_plan(p) -> str:
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(p.steps))
    return f"**{p.fab_method}** · {len(p.primitives)} primitives\n{steps}"


def _fmt_stats(st: dict[str, Any]) -> str:
    return (f"watertight={st['watertight']} bbox={st['bbox_mm']}mm faces={st['n_faces']}"
            + (f" vol={st['volume_mm3']}mm³" if st.get('volume_mm3') else ""))


def _fmt_quality(q: dict[str, Any]) -> str:
    parts = []
    if "dimension_score" in q:
        parts.append(f"**dimension match: {q['dimension_score']:.0%}** "
                     f"(got {q['dims_got_mm']} vs target {q['dims_target_mm']} mm)")
    if "chamfer" in q:
        parts.append(f"Chamfer={q['chamfer']} · voxel IoU={q.get('voxel_iou')}")
    if "silhouette_iou" in q:
        vp = f" (best view {q['viewpoint']})" if q.get("viewpoint") else ""
        parts.append(f"**silhouette match vs photo: {q['silhouette_iou']:.0%}**{vp}")
    if "reference_error" in q:
        parts.append(f"(reference compare failed: {q['reference_error']})")
    return "\n".join(parts) or "no ground truth available"
