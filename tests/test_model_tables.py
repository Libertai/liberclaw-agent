"""Pins the per-model facts the agent acts on: which model ids take images
directly, and the compaction budget each one gets.

Imports only ``baal_agent.compaction`` (not ``main``), so it runs without
FastAPI installed.
"""

from __future__ import annotations

import pytest

from baal_agent.compaction import (
    CONTEXT_BUDGET_TOKENS,
    get_context_limit,
    model_supports_vision,
)

BRANDED_ALIASES = ["claw-flash", "claw-core", "deep-claw", "claw-large"]


@pytest.mark.parametrize("model", BRANDED_ALIASES)
def test_branded_aliases_accept_images(model):
    assert model_supports_vision(model) is True


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3-coder-next", True),
        ("glm-4.7", True),
        ("hermes-3-8b-tee", False),
    ],
)
def test_legacy_ids_keep_their_vision_flag(model, expected):
    assert model_supports_vision(model) is expected


def test_unknown_model_is_assumed_to_accept_images():
    assert model_supports_vision("some-future-model") is True


def test_budget_constant_is_200k():
    assert CONTEXT_BUDGET_TOKENS == 200_000


@pytest.mark.parametrize("model", BRANDED_ALIASES)
def test_branded_aliases_share_the_compaction_budget(model):
    # A budget, not the advertised window: these models advertise 262K to 1M.
    assert get_context_limit(model, 0) == 200_000


def test_configured_max_overrides_the_budget():
    assert get_context_limit("claw-large", 50_000) == 50_000
