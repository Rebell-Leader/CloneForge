# ⚒️ CloneForge

**Real-time multimodal object-cloning agent swarm — Gemma 4 31B on Cerebras.**

Upload or webcam a photo of a physical object; a swarm of specialized agents analyzes it,
plans a fabrication strategy, generates parametric 3D code, **visually critiques its own
result against your photo**, and emits a watertight, 3D-printable STL — in seconds, thanks
to Cerebras inference (~1,500+ tok/s).

```
photo(s) ─► 👁 Vision ─► 🧠 Planner ─► 🛠 Generator ─► 🔎 Visual Critic ─┐
              (specs)     (primitives)   (trimesh code)   (render vs photo) │
                                  ▲                                         │
                                  └──────── fix & regenerate (≤N) ◄─────────┘
                                                   │
                                      watertight STL + 3D preview + quality report
```

## Why it's different
- **The critic has eyes.** After building a mesh we render it (4 views) and send that render
  *back to Gemma alongside the original photo* — the model sees the mismatch and issues concrete
  fixes. This render→VLM→fix loop is the main fidelity lever (cf. Query2CAD, CADCodeVerify, LL3M).
- **Watertight by construction.** Output is a composition of parametric primitives
  (box/cylinder/sphere/torus + booleans), so meshes are print-ready with no repair pass —
  unlike neural image→3D models (TripoSR/TRELLIS/Hunyuan3D) that need GPUs and produce
  non-manifold draft meshes. It's also **editable**: ask for "20% taller" and it re-runs.
- **Speed is the demo.** A full clone (8–12 agent calls) runs in ~5 s of compute. The
  Speed Race tab shows Cerebras vs OpenAI side-by-side with live TTFT + tok/s.

## Quickstart
```bash
# Python 3.14, uv (python -m venv is unavailable here; uv handles the 3.14 wheels)
uv venv --python 3.14 .venv && . .venv/bin/activate
uv pip install -r requirements.txt

# .env needs:  CEREBRAS_API_KEY=...   OPENAI_API_KEY=...   (OpenAI = fallback + race lane)
python app.py        # open the printed local URL
```

## The app (3 tabs)
- **Clone** — photo (upload/webcam) + optional extra views → live agent transcript, mesh render,
  3D preview, downloadable STL. A **Refine** box applies text corrections ("thinner handle")
  reusing the cached analysis.
- **Examples** — curated reference objects with **published ground truth** (LEGO 3001, ISO 7089
  washer, DIN 934 nut, 16 mm die, mug, 20-tooth gear). One click clones them and reports
  **dimension match %** and Chamfer/voxel-IoU. Simple parts score high; the gear shows the
  fidelity gap on complex geometry — honest by design.
- **Speed Race** — same prompt, Cerebras ⚡ vs OpenAI 🐢, live first-token latency + tok/s.

## Architecture
| Module | Role |
|---|---|
| `cloneforge/llm.py` | Unified `AsyncOpenAI` client for **both** providers (Cerebras is OpenAI-compatible); streaming, multi-image input, strict JSON schema, 30-rpm backoff → OpenAI fallback |
| `cloneforge/schemas.py` | Pydantic agent I/O → strict `json_schema` (`strictify`) |
| `cloneforge/agents.py` | Vision · Planner · Generator · **Visual Critic** (zero tool-calling — see below) |
| `cloneforge/fabricate.py` | Sandboxed exec of generated code (whitelisted imports) + STL/GLB + watertight validation + stderr self-repair |
| `cloneforge/render.py` | Headless matplotlib 4-view shaded render (no GPU/X11/sudo) |
| `cloneforge/quality.py` | OBB dimension match + Chamfer + voxel-IoU vs ground truth |
| `cloneforge/silhouette.py` | Silhouette-IoU vs the input photo (best-of-N ranking + shape-match score) |
| `cloneforge/orchestrator.py` | Async-generator pipeline (streams to UI) + `refine_pipeline` |
| `cloneforge/examples.py` | Reference library from published part specs |
| `app.py` | Gradio UI |

## Key technical decisions (verified against the live API)
- **Images cannot be combined with tool calling** on Gemma 4 → we use **structured outputs
  everywhere, zero tool calling.**
- **`reasoning_effort` levels are equivalent** on Gemma 4 and *destabilize* structured output
  (empty JSON) → kept **off** on schema'd agents.
- **30 rpm** rate limit → vision is computed once and cached across critic/refine iterations;
  webcam is snapshot-only; 429 → bounded backoff → OpenAI `gpt-5.4-mini` fallback.
- **Python 3.14, no sudo** → CadQuery (≤3.12) and OpenSCAD (apt) are out; **trimesh + manifold3d**
  (pure pip) is the generator, **matplotlib** the renderer.

See [PLAN.md](PLAN.md) for the full build plan and [FIDELITY.md](FIDELITY.md) for the fidelity
analysis, competitive landscape, and roadmap. Demo script: [DEMO.md](DEMO.md).
