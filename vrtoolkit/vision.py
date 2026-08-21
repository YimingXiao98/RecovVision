"""
OpenAI Vision/Text integration helpers for the GPT-4o pipeline.

Prompts here are the *canonical* ones reported in the paper appendix:
  - the 9-key snake_case attribute-extraction schema (Listing 1), and
  - the few-shot decision prompt with 9 exemplars (Listing 3).

This is the prompt-matched reproduction used for the open-weight comparison.
The original GPT-4o run phrased each attribute as a natural-language question
and, in particular, excluded the survey vehicle in its vehicle attribute; that
one wording difference is documented in the paper (Section 4.3) and does not
affect the eight non-vehicle cues.

No API keys are embedded; set OPENAI_API_KEY in the environment before use.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Tuple

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional
    openai = None

DEFAULT_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
DEFAULT_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o")

# Canonical 9-attribute schema (order matters; matches the paper).
NINE = [
    "house_destruction", "structural_damage", "exterior_debris", "open_doors_windows",
    "site_accessible", "exterior_mud", "emergency_markings", "major_repairs", "vehicle_presence",
]

VISION_PROMPT = (
    "Analyze the image and answer with a JSON object using EXACTLY these keys and boolean "
    "values (true/false). Do not add, remove, or rename keys. Return only the JSON object, "
    "no prose.\n\n{\n" + ",\n".join(f'  "{k}": true/false' for k in NINE) + "\n}"
)

# Few-shot exemplars for the decision stage (snake_case), matching Appendix Listing 3.
_FEWSHOT = [
    ({"site_accessible": True, "vehicle_presence": True}, "Occupied"),
    ({"house_destruction": True, "structural_damage": True, "exterior_debris": True,
      "site_accessible": True}, "Not Occupied"),
    ({"structural_damage": True, "exterior_debris": True, "open_doors_windows": True,
      "site_accessible": True, "vehicle_presence": True}, "Not Occupied"),
    ({"structural_damage": True, "site_accessible": True, "major_repairs": True}, "Not Occupied"),
    ({"site_accessible": True, "emergency_markings": True}, "Not Occupied"),
    ({"site_accessible": True}, "Occupied"),
    ({"house_destruction": True, "site_accessible": True}, "Not Occupied"),
    ({"site_accessible": False}, "Not Occupied"),
    ({"site_accessible": True, "exterior_mud": True, "vehicle_presence": True}, "Occupied"),
]


def _full(attrs: dict) -> dict:
    """Fill all 9 keys, defaulting missing ones to False."""
    return {k: bool(attrs.get(k, False)) for k in NINE}


def build_decision_prompt(vision_json_str: str) -> str:
    head = (
        "You are an expert in post-disaster building occupancy assessment. Given a building's "
        "attributes in JSON (keys listed below), decide if it is 'Occupied' or 'Not Occupied'.\n\n"
        "Consider all evidence: for example, some exterior mud may coexist with occupancy; parked "
        "vehicles can indicate occupancy when other signs are mixed; extensive roof or wall repairs "
        "may indicate temporary non-occupancy. If the evidence is mixed or unclear, prefer 'Not "
        "Occupied'. Be conservative: if there is significant indication that a building might not be "
        "occupied (e.g., destruction, inaccessibility, visible abandonment, or multiple risk "
        "factors), classify it as 'Not Occupied'. Classify as 'Occupied' only if the building "
        "appears livable and there are no clear signs of uninhabitability.\n\n"
        "The JSON fields you will receive:\n" + json.dumps({k: "bool" for k in NINE}, indent=2) +
        "\n\nHere are some examples:\n"
    )
    body = ""
    for i, (a, lab) in enumerate(_FEWSHOT, 1):
        body += f"\nExample {i}:\n{json.dumps(_full(a), indent=2)}\n{lab}\n"
    tail = (f"\nNow, decide for this building:\n{vision_json_str}\n\n"
            "Output only one token: 'Occupied' or 'Not Occupied'.")
    return head + body + tail


def _client():
    if openai is None:
        raise RuntimeError("openai package is not installed. `pip install openai`.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    openai.api_key = api_key


def call_vision_model(image_path: str, model: str = DEFAULT_VISION_MODEL) -> Tuple[str, int | None]:
    """Extract the 9 boolean attributes from one facade image; returns (json_str, total_tokens)."""
    _client()
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output only strict JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        max_tokens=200,
        temperature=0,
    )
    usage = getattr(getattr(resp, "usage", None), "total_tokens", None)
    return resp.choices[0].message.content, usage


def call_text_model(vision_output_str: str, model: str = DEFAULT_TEXT_MODEL) -> Tuple[str, int | None]:
    """Two-stage decision: map extracted attributes to a label via the few-shot prompt."""
    _client()
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ("You are an expert in post-disaster building occupancy "
                                           "assessment. Output only one token: Occupied or Not Occupied.")},
            {"role": "user", "content": build_decision_prompt(vision_output_str)},
        ],
        max_tokens=5,
        temperature=0,
    )
    out = (resp.choices[0].message.content or "").strip()
    lab = "Not Occupied" if "not occupied" in out.lower() else ("Occupied" if "occupied" in out.lower() else out)
    usage = getattr(getattr(resp, "usage", None), "total_tokens", None)
    return lab, usage
