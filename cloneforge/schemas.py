"""Agent I/O schemas (Pydantic) + helpers to emit OpenAI strict json_schema.

Strict mode requires every object to set additionalProperties:false and list all
properties as required. `strictify` enforces that recursively over a Pydantic schema.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Vision agent
# ---------------------------------------------------------------------------
class Dimensions(BaseModel):
    height_mm: float
    width_mm: float
    depth_mm: float


class VisionSpec(BaseModel):
    object: str = Field(description="Concise name of the primary object")
    geometry: str = Field(description="Geometric description (shape, parts, proportions)")
    dimensions: Dimensions = Field(description="Best-estimate real-world size in mm")
    materials: list[str]
    features: list[str] = Field(description="Salient features: handles, holes, ribs, text")
    defects: list[str] = Field(description="Visible damage/defects, empty if none")
    confidence: float = Field(description="0..1 confidence in the analysis")


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------
class Primitive(BaseModel):
    type: Literal["box", "cylinder", "sphere", "torus"]
    op: Literal["add", "subtract"] = Field(description="Boolean op vs the accumulated body")
    dims_mm: list[float] = Field(description="box:[x,y,z] cyl:[r,h] sphere:[r] torus:[R,r]")
    pose_mm: list[float] = Field(description="[x,y,z] translation of this primitive's center")


class FabPlan(BaseModel):
    fab_method: str = Field(description="e.g. '3D printing (FDM)', 'CNC', 'repair'")
    steps: list[str] = Field(description="Human-readable fabrication steps")
    primitives: list[Primitive] = Field(description="Constructive primitives, applied in order")
    notes: str


# ---------------------------------------------------------------------------
# Generator agent
# ---------------------------------------------------------------------------
class GeneratedArtifact(BaseModel):
    lang: Literal["trimesh"] = "trimesh"
    code: str = Field(description="Python using `trimesh` and `np`; must assign `result`")
    est_bbox_mm: list[float] = Field(description="Estimated bounding box [x,y,z]")


# ---------------------------------------------------------------------------
# Critic agent
# ---------------------------------------------------------------------------
class CritiqueVerdict(BaseModel):
    approved: bool
    score: float = Field(description="0..1 fidelity vs the analyzed object")
    issues: list[str]
    fix_instructions: str = Field(description="Concrete changes for the generator, empty if approved")


def strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively make a JSON schema strict-mode compatible."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
        for v in schema.values():
            strictify(v)
    elif isinstance(schema, list):
        for v in schema:
            strictify(v)
    return schema


def response_format(model: type[BaseModel]) -> dict:
    """Build a Cerebras/OpenAI strict json_schema response_format from a Pydantic model."""
    schema = strictify(model.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__, "strict": True, "schema": schema},
    }
