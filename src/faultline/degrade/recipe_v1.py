"""Frozen degradation recipe v1 (see configs/degradation_v1.yaml). Same parameters as the provisional recipe."""

from __future__ import annotations

import hashlib
import json

from .provisional import PROVISIONAL_RECIPE

RECIPE_V1 = {**PROVISIONAL_RECIPE, "name": "degradation_v1"}


def recipe_hash(recipe: dict = RECIPE_V1) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest()
