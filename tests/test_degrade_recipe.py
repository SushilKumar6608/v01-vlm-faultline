"""The frozen degradation recipe must not drift: yaml, code and pinned hash all agree."""
from pathlib import Path

import yaml

from faultline.degrade.provisional import PROVISIONAL_RECIPE, noise_seed
from faultline.degrade.recipe_v1 import RECIPE_V1, recipe_hash

PINNED_HASH = "23356dcd8b203486df22394109e9008b50b18d1be55998168e6934208b9d528c"
CFG = Path(__file__).resolve().parents[1] / "configs" / "degradation_v1.yaml"


def test_pinned_hash():
    assert recipe_hash(RECIPE_V1) == PINNED_HASH


def test_yaml_matches_code():
    y = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    for k, v in RECIPE_V1.items():
        got = tuple(y[k]) if isinstance(v, tuple) else y[k]
        assert got == v, k


def test_same_parameters_as_provisional():
    a = {k: v for k, v in RECIPE_V1.items() if k != "name"}
    b = {k: v for k, v in PROVISIONAL_RECIPE.items() if k != "name"}
    assert a == b


def test_noise_seed_rule_unchanged():
    assert noise_seed(950000) == (950000 * 7919 + 13) % 2_147_483_647
