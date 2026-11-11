from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pytest

from v3_chan.explicit_feedback.online_config import (
    DEFAULT_PROTOCOL_CONFIG,
    DEFAULT_SCENARIO_CONFIG,
    load_json_object,
    validate_frozen_protocol_config,
)


_PINNED_PROTOCOL_GROUPS = (
    "protocol_version",
    "policy",
    "task",
    "cbf",
    "feedback",
    "encounter",
    "design",
)
_PINNED_SCENARIO_GROUPS = (
    "schema_version",
    "surface_gap_reference",
    "severity",
    "crossing",
    "nominal_bc_reference",
    "layout_precheck",
    "protected_crossing",
)


def _leaf_paths(value: Mapping[str, Any], prefix: tuple[str, ...] = ()):
    for key, item in value.items():
        path = (*prefix, str(key))
        if isinstance(item, Mapping):
            yield from _leaf_paths(item, path)
        else:
            yield path


def _mutated(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.001
    if isinstance(value, str):
        return value + "_mutated"
    if isinstance(value, list):
        return [*value, "__mutated__"]
    raise AssertionError(f"test has no mutation strategy for {type(value).__name__}")


def _replace_leaf(root: dict[str, Any], path: tuple[str, ...]) -> None:
    target: dict[str, Any] = root
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = _mutated(target[path[-1]])


def _cases(config_path, groups: tuple[str, ...], label: str):
    config = load_json_object(config_path)
    selected = {name: config[name] for name in groups}
    return [
        pytest.param(label, path, id=f"{label}-{'-'.join(path)}")
        for path in _leaf_paths(selected)
    ]


_MUTATION_CASES = [
    *_cases(DEFAULT_PROTOCOL_CONFIG, _PINNED_PROTOCOL_GROUPS, "protocol"),
    *_cases(DEFAULT_SCENARIO_CONFIG, _PINNED_SCENARIO_GROUPS, "scenario"),
]


def test_shipped_online_configs_satisfy_frozen_contract() -> None:
    protocol = load_json_object(DEFAULT_PROTOCOL_CONFIG)
    scenario = load_json_object(DEFAULT_SCENARIO_CONFIG)

    validate_frozen_protocol_config(protocol, scenario)


def test_boolean_contract_values_do_not_accept_integer_aliases() -> None:
    protocol = load_json_object(DEFAULT_PROTOCOL_CONFIG)
    scenario = load_json_object(DEFAULT_SCENARIO_CONFIG)
    protocol["policy"]["frozen"] = 1

    with pytest.raises(ValueError, match="frozen protocol mismatch at policy"):
        validate_frozen_protocol_config(protocol, scenario)


@pytest.mark.parametrize(("target_name", "path"), _MUTATION_CASES)
def test_every_behavioral_config_leaf_is_frozen(
    target_name: str, path: tuple[str, ...]
) -> None:
    protocol = load_json_object(DEFAULT_PROTOCOL_CONFIG)
    scenario = load_json_object(DEFAULT_SCENARIO_CONFIG)
    target = deepcopy(protocol if target_name == "protocol" else scenario)
    _replace_leaf(target, path)

    with pytest.raises(ValueError, match="frozen (protocol|scenario) mismatch"):
        validate_frozen_protocol_config(
            target if target_name == "protocol" else protocol,
            target if target_name == "scenario" else scenario,
        )
