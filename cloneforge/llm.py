"""Unified LLM client.

Cerebras is OpenAI-compatible, so a single `AsyncOpenAI` code path drives both the
fast Cerebras/Gemma lane and the OpenAI fallback/race lane. Cerebras-only params
(`reasoning_effort`) go through `extra_body`.

Verified constraints (see PLAN.md §0):
  - gemma-4-31b: image inputs (base64 data-URI only), strict json_schema, reasoning_effort.
  - Images CANNOT be combined with tool calling -> we use structured outputs only.
  - Rate limit ~30 rpm -> on RateLimitError we fall back to OpenAI gpt-5.4-mini.
"""
from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError, RateLimitError

load_dotenv()

CEREBRAS_MODEL = "gemma-4-31b"
OPENAI_FALLBACK_MODEL = "gpt-5.4-mini"   # multimodal; covers the vision agent on fallback
OPENAI_RACE_MODEL = "gpt-5.4-mini"       # "slow" lane for the speed race

_cerebras = AsyncOpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)
_openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROVIDERS = {
    "cerebras": (_cerebras, CEREBRAS_MODEL),
    "openai": (_openai, OPENAI_RACE_MODEL),
}

RATE_LIMIT_RETRIES = 2

# Optional UI hook: register a callback to surface rate-limit / fallback status live.
_status_hook: Callable[[str], None] | None = None


def set_status_hook(fn: Callable[[str], None] | None) -> None:
    global _status_hook
    _status_hook = fn


def _emit_status(msg: str) -> None:
    if _status_hook:
        try:
            _status_hook(msg)
        except Exception:  # noqa: BLE001 — status is best-effort
            pass


def _retry_after(exc: Exception, attempt: int) -> float:
    """Seconds to wait: honor Retry-After header if present, else exp backoff."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return min(float(resp.headers.get("retry-after")), 10.0)
        except (TypeError, ValueError):
            pass
    return min(1.2 * (2 ** attempt), 8.0)


@dataclass
class CallMeta:
    provider: str
    model: str
    latency_s: float
    completion_tokens: int = 0
    fell_back: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def encode_image(path_or_bytes: str | bytes, fmt: str = "PNG") -> str:
    """Return a base64 data-URI for an image path or raw bytes (PNG/JPEG only)."""
    if isinstance(path_or_bytes, str):
        with open(path_or_bytes, "rb") as f:
            raw = f.read()
        ext = path_or_bytes.rsplit(".", 1)[-1].lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    else:
        raw = path_or_bytes
        mime = "png" if fmt.upper() == "PNG" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode()}"


def image_content(text: str, data_uri: str) -> list[dict]:
    """Build a multimodal user-message content list (text + one image)."""
    return multi_image_content(text, [data_uri])


def multi_image_content(text: str, data_uris: list[str], labels: list[str] | None = None) -> list[dict]:
    """Build a multimodal content list: text + up to 5 images (Gemma 4 limit).

    Optional per-image labels are inserted as text so the model knows which view is which.
    """
    data_uris = data_uris[:5]  # Gemma 4: max 5 images/request
    parts: list[dict] = [{"type": "text", "text": text}]
    for i, uri in enumerate(data_uris):
        if labels and i < len(labels):
            parts.append({"type": "text", "text": f"[{labels[i]}]"})
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


# ---------------------------------------------------------------------------
# non-streaming call (used by the agent pipeline) with rate-limit fallback
# ---------------------------------------------------------------------------
async def acall(
    messages: list[dict],
    *,
    schema: dict | None = None,
    reasoning: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    fallback: bool = True,
) -> tuple[str, CallMeta]:
    """Call Cerebras/Gemma; on rate-limit or transient error fall back to OpenAI.

    `schema` is a strict json_schema dict (see schemas.response_format).
    """
    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    if schema is not None:
        kwargs["response_format"] = schema

    # primary: Cerebras
    extra = {"reasoning_effort": reasoning} if reasoning else None

    # primary: Cerebras, with bounded backoff on rate limits (30 rpm budget)
    last_exc: Exception | None = None
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            resp = await _cerebras.chat.completions.create(
                model=CEREBRAS_MODEL, extra_body=extra, **kwargs
            )
            text, meta = _finish(resp, "cerebras", CEREBRAS_MODEL, t0, fell_back=False)
            meta.extra["retries"] = attempt
            return text, meta
        except RateLimitError as e:
            last_exc = e
            if attempt < RATE_LIMIT_RETRIES:
                delay = _retry_after(e, attempt)
                _emit_status(f"⏳ Cerebras rate-limited — retrying in {delay:.1f}s "
                             f"({attempt + 1}/{RATE_LIMIT_RETRIES})")
                await asyncio.sleep(delay)
                continue
            break
        except APIStatusError as e:
            last_exc = e
            break

    # fallback: OpenAI (drop Cerebras-only reasoning_effort)
    if not fallback:
        raise last_exc  # type: ignore[misc]
    _emit_status("↪ Falling back to OpenAI gpt-5.4-mini")
    t0 = time.perf_counter()
    resp = await _openai.chat.completions.create(model=OPENAI_FALLBACK_MODEL, **kwargs)
    meta = _meta(resp, "openai", OPENAI_FALLBACK_MODEL, t0, fell_back=True)
    meta.extra["fallback_reason"] = type(last_exc).__name__ if last_exc else "unknown"
    return resp.choices[0].message.content or "", meta


def _meta(resp, provider, model, t0, fell_back) -> CallMeta:
    usage = getattr(resp, "usage", None)
    return CallMeta(
        provider=provider,
        model=model,
        latency_s=time.perf_counter() - t0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        fell_back=fell_back,
    )


def _finish(resp, provider, model, t0, fell_back):
    return resp.choices[0].message.content or "", _meta(resp, provider, model, t0, fell_back)


# ---------------------------------------------------------------------------
# streaming call (used by the Speed Race tab) with live TTFT + tok/s
# ---------------------------------------------------------------------------
async def astream(
    provider: str, prompt: str, *, max_tokens: int = 800
) -> AsyncIterator[tuple[str, dict]]:
    """Yield (accumulated_text, stats) as tokens arrive. stats: ttft_ms, tok_s, elapsed_s."""
    client, model = PROVIDERS[provider]
    t0 = time.perf_counter()
    ttft: float | None = None
    n = 0
    acc = ""
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if not delta:
            continue
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
        acc += delta
        elapsed = time.perf_counter() - t0
        yield acc, {
            "provider": provider,
            "model": model,
            "ttft_ms": (ttft or 0) * 1000,
            "tok_s": n / elapsed if elapsed else 0,
            "elapsed_s": elapsed,
            "tokens": n,
        }
