"""Compose a Hydra config and extract its parallelism/model fields.

Hydra configs are rarely one file: a root config's `defaults:` list names
other YAML files (config groups) that get deep-merged in, with the root's
own keys (`_self_`) taking their place in that ordering - and CLI
`key=value` overrides applying on top of all of it. Getting tp/pp/dp/model
right means actually doing that composition, not just reading the root
file.
"""

from pathlib import Path
from typing import Any, Optional

import yaml

SELF_MARKER = "_self_"


def extract_hydra(config_path: str, overrides: Optional[list] = None) -> dict:
    """Compose a Hydra config (root + its `defaults:` chain + overrides)
    and return the parallelism/model fields relevant to traincheck.
    """
    root_path = Path(config_path)
    config_dir = root_path.parent
    root_doc = _load_yaml(root_path)

    merged = _resolve_defaults(root_doc, config_dir)
    merged = _apply_overrides(merged, overrides or [])

    parallelism = merged.get("parallelism") or {}
    return {
        "tensor_parallel": parallelism.get("tensor_parallel"),
        "pipeline_parallel": parallelism.get("pipeline_parallel"),
        "data_parallel": parallelism.get("data_parallel"),
        "sharding": parallelism.get("sharding"),
        "model": merged.get("model"),
    }


def _resolve_defaults(root_doc: dict, config_dir: Path) -> dict:
    merged: dict = {}
    self_applied = False

    for entry in root_doc.get("defaults") or []:
        if entry == SELF_MARKER:
            merged = _deep_merge(merged, _own_keys(root_doc))
            self_applied = True
            continue
        if isinstance(entry, dict):
            for group, option in entry.items():
                group_doc = _load_yaml(config_dir / group / f"{option}.yaml")
                merged = _deep_merge(merged, group_doc)

    if not self_applied:
        # Hydra's own default ordering: the primary config's keys win over
        # its defaults unless `_self_` was explicitly placed earlier.
        merged = _deep_merge(merged, _own_keys(root_doc))

    return merged


def _own_keys(doc: dict) -> dict:
    return {key: value for key, value in doc.items() if key != "defaults"}


def _deep_merge(base: dict, other: dict) -> dict:
    merged = dict(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_overrides(merged: dict, overrides: list) -> dict:
    for override in overrides:
        if "=" not in override:
            continue
        dotted_key, raw_value = override.split("=", 1)
        _set_dotted(merged, dotted_key.strip(), _coerce(raw_value.strip()))
    return merged


def _set_dotted(doc: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = doc
    for part in parts[:-1]:
        if not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _coerce(raw_value: str) -> Any:
    """Hydra overrides are YAML-scalar-typed on the CLI: "4" -> int,
    "true" -> bool, anything else stays a plain string.
    """
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _load_yaml(path: Path) -> dict:
    try:
        text = path.read_text()
    except OSError:
        return {}
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return doc if isinstance(doc, dict) else {}
