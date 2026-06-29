"""CloneForge — Gradio app.

Tab 1 (Clone): image/webcam (+ optional extra views) -> agent swarm -> live
  transcript + multi-view mesh render + 3D preview + STL + quality vs ground truth.
Tab 2 (Examples): curated reference objects with published ground truth; one click
  clones them and reports dimension/Chamfer accuracy (and the gap on complex parts).
Tab 3 (Speed Race): same prompt on Cerebras vs OpenAI, live TTFT + tok/s.

Run:  python app.py
"""
from __future__ import annotations

import sys

import gradio as gr

from cloneforge import examples, llm
from cloneforge.orchestrator import clone_pipeline, refine_pipeline

EXAMPLE_GOAL = "Clone this object as a 3D-printable model"

llm.set_status_hook(lambda m: print(f"[status] {m}", file=sys.stderr))
MANIFEST = examples.load_manifest()


def _uris(main_image, extra_files):
    uris = []
    if main_image:
        uris.append(llm.encode_image(main_image))
    for f in (extra_files or []):
        uris.append(llm.encode_image(f))
    return uris[:5]  # Gemma 4 limit


def _fmt_quality(q):
    if not q:
        return ""
    rows = []
    if "dimension_score" in q:
        rows.append(f"| Dimension match | **{q['dimension_score']:.0%}** "
                    f"(got {q['dims_got_mm']} vs {q['dims_target_mm']} mm) |")
    if "chamfer" in q:
        rows.append(f"| Chamfer (↓) | {q['chamfer']} |")
        rows.append(f"| Voxel IoU (↑) | {q.get('voxel_iou')} |")
    if not rows:
        return ""
    return "### 📊 Accuracy vs ground truth\n| metric | value |\n|---|---|\n" + "\n".join(rows)


def _outputs(st, n_uris=0, extra_summary=""):
    summary = f"**{st.total_calls} agent calls · {st.total_latency_s:.2f}s compute** {extra_summary}"
    return (st.transcript, st.render_png, st.glb_path, (st.code or ""),
            st.stl_path, _fmt_quality(st.quality), summary, st)


async def run_clone(main_image, extra_files, goal, ex):
    uris = _uris(main_image, extra_files)
    if not uris:
        yield ([{"role": "assistant", "content": "Add a photo (upload/webcam) first."}],
               None, None, "", None, "", "", None)
        return
    goal = goal or EXAMPLE_GOAL
    target = ex.get("dims_mm") if ex else None
    ref = ex.get("reference_stl") if ex else None
    async for st in clone_pipeline(uris, goal, target_dims_mm=target, reference_mesh=ref):
        yield _outputs(st, len(uris), f"({len(uris)} view{'s' if len(uris) > 1 else ''})")


async def run_refine(state, instruction, ex):
    if state is None:
        yield ([{"role": "assistant", "content": "Clone something first, then refine."}],
               None, None, "", None, "", "", None)
        return
    if not (instruction or "").strip():
        yield _outputs(state)
        return
    target = ex.get("dims_mm") if ex else None
    ref = ex.get("reference_stl") if ex else None
    async for st in refine_pipeline(state, instruction, target_dims_mm=target, reference_mesh=ref):
        yield _outputs(st, extra_summary="(refined)")


def _lane(provider: str):
    async def handler(prompt):
        prompt = prompt or "Explain how a 3D printer extrudes filament, in 5 sentences."
        async for acc, stats in llm.astream(provider, prompt):
            md = (f"**{stats['provider']}** · `{stats['model']}`  \n"
                  f"⏱ TTFT **{stats['ttft_ms']:.0f} ms** · "
                  f"🚀 **{stats['tok_s']:.0f} tok/s** · {stats['elapsed_s']:.2f}s")
            yield [{"role": "assistant", "content": acc}], md
    return handler


def build_ui():
    with gr.Blocks(title="CloneForge") as demo:
        gr.Markdown("# ⚒️ CloneForge\n"
                    "Real-time multimodal object-cloning agent swarm — **Gemma 4 31B on Cerebras**. "
                    "Photo → vision → plan → generate → *visual* critique → printable STL.")
        ex_state = gr.State(None)
        clone_state = gr.State(None)

        with gr.Tab("Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(label="Object photo", sources=["upload", "webcam"], type="filepath")
                    extra = gr.File(label="Extra views (optional: side/top)",
                                    file_count="multiple", file_types=["image"], type="filepath")
                    goal = gr.Textbox(label="Goal", value=EXAMPLE_GOAL)
                    run_btn = gr.Button("⚡ Clone it", variant="primary")
                    summary = gr.Markdown()
                    quality = gr.Markdown()
                    gr.Markdown("**Refine** — ask for a correction (reuses the analysis):")
                    refine_box = gr.Textbox(show_label=False, placeholder="e.g. make it 20% taller / thinner handle")
                    refine_btn = gr.Button("🔁 Refine")
                with gr.Column(scale=1):
                    chat = gr.Chatbot(label="Agent swarm", height=460)
                with gr.Column(scale=1):
                    render = gr.Image(label="Model render (what the critic sees)", height=240)
                    model3d = gr.Model3D(label="3D preview", display_mode="solid", height=240)
                    stl = gr.File(label="Download STL")
            with gr.Accordion("Generated code", open=False):
                code = gr.Code(language="python")
            outs = [chat, render, model3d, code, stl, quality, summary, clone_state]
            run_btn.click(run_clone, [img, extra, goal, ex_state], outs)
            refine_btn.click(run_refine, [clone_state, refine_box, ex_state], outs)

        with gr.Tab("Examples"):
            gr.Markdown("### Reference objects with published ground truth\n"
                        "Simple parts clone accurately (numeric dimension match); the **gear** shows "
                        "the fidelity gap on complex geometry. Click one to load it, then **Clone it**.")
            gallery = gr.Gallery(
                value=[(it["preview"], f"{it['title']} · {it['category']}") for it in MANIFEST],
                columns=3, height=320, allow_preview=False, label="Click to load")
            ex_note = gr.Markdown()

            def pick(evt: gr.SelectData):
                it = MANIFEST[evt.index]
                return it["preview"], it["goal"], it, f"**{it['title']}** — ground truth: {it['note']}"
            gallery.select(pick, None, [img, goal, ex_state, ex_note])

        with gr.Tab("Speed Race"):
            gr.Markdown("### Same prompt, two providers — watch the first token land.")
            prompt = gr.Textbox(label="Prompt",
                                value="Explain how a 3D printer extrudes filament, in 5 sentences.")
            race_btn = gr.Button("🏁 Race", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚡ Cerebras · Gemma 4 31B")
                    cb_stat = gr.Markdown()
                    cb_chat = gr.Chatbot(height=320, show_label=False)
                with gr.Column():
                    gr.Markdown("### 🐢 OpenAI · gpt-5.4-mini")
                    oa_stat = gr.Markdown()
                    oa_chat = gr.Chatbot(height=320, show_label=False)
            race_btn.click(_lane("cerebras"), prompt, [cb_chat, cb_stat], concurrency_limit=None)
            race_btn.click(_lane("openai"), prompt, [oa_chat, oa_stat], concurrency_limit=None)

    demo.queue(default_concurrency_limit=None)
    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft())
