"""Compose a Hydra config and extract its parallelism/model fields.

Hydra configs are rarely one file: a root config's `defaults:` list names
other YAML files (config groups) that get deep-merged in, with the root's
own keys (`_self_`) taking their place in that ordering - and CLI
overrides applying on top of all of it. Getting tp/pp/dp/model right means
actually doing that composition, not just reading the root file.

CLI overrides come in two unrelated shapes that share the same
`key=value` spelling:
- a GROUP override (`model=llama`) swaps which file backs an entire
  group - "load configs/model/llama.yaml in place of whatever `model:`
  default was already composed in" - so its right-hand side is a
  filename, not a value to store.
- a VALUE override (`trainer.tensor_parallel=4`, or a bare key that
  doesn't name a group) sets a value at that path, same as before.

Group overrides are detected by checking the key against the known group
names (the defaults list's own group keys, plus any config subdirectory)
rather than by the presence of a dot, since a group key is always a
single flat name - and they're applied before value overrides, matching
Hydra's own order: group overrides affect what gets composed, value
overrides then apply on top of the fully-composed result.
"""

import re
from pathlib import Path
from typing import Any, Optional

import yaml

SELF_MARKER = "_self_"

_INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")


def extract_hydra(config_path: str, overrides: Optional[list] = None) -> dict:
    """Compose a Hydra config (root + its `defaults:` chain + overrides)
    and return the parallelism/model fields relevant to traincheck.
    """
    root_path = Path(config_path)
    config_dir = root_path.parent
    root_doc = _load_yaml(root_path)

    merged = _resolve_defaults(root_doc, config_dir)
    groups = _known_groups(root_doc, config_dir)
    merged = _apply_overrides(merged, overrides or [], config_dir, groups)
    merged = _resolve_interpolations(merged)

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


def _known_groups(root_doc: dict, config_dir: Path) -> set:
    """Every name a `group=option` override could plausibly refer to: the
    defaults list's own group keys, plus any actual config subdirectory
    (covering a group override for a group not already in `defaults:`).
    """
    groups = set()
    for entry in root_doc.get("defaults") or []:
        if isinstance(entry, dict):
            groups.update(entry.keys())
    if config_dir.is_dir():
        groups.update(p.name for p in config_dir.iterdir() if p.is_dir())
    return groups


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


def _apply_overrides(merged: dict, overrides: list, config_dir: Path, groups: set) -> dict:
    group_overrides = []
    value_overrides = []

    for override in overrides:
        if "=" not in override:
            continue
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if "." not in key and key in groups:
            group_overrides.append((key, raw_value.strip()))
        else:
            value_overrides.append((key, raw_value.strip()))

    # Group overrides first - they replace what got composed in from
    # `defaults:`, which value overrides then apply on top of.
    for group, option in group_overrides:
        group_doc = _load_yaml(config_dir / group / f"{option}.yaml")
        merged.pop(group, None)
        merged = _deep_merge(merged, group_doc)

    for dotted_key, raw_value in value_overrides:
        _set_dotted(merged, dotted_key, _coerce(raw_value))

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


def _resolve_interpolations(doc: dict) -> dict:
    """Resolve OmegaConf-style `${a.b}` references against the same
    composed document. Only plain dotted-path self-references are
    supported - not resolver functions like `${oc.env:...}`, which Hydra
    configs can use but traincheck's own fields never need.
    """
    return _resolve_value(doc, doc, frozenset())


def _resolve_value(doc: dict, value: Any, seen: frozenset) -> Any:
    if isinstance(value, str):
        match = _INTERPOLATION_RE.fullmatch(value)
        if match:
            resolved = _lookup(doc, match.group(1).strip(), seen)
            return value if resolved is None else resolved

        def substitute(m: re.Match) -> str:
            resolved = _lookup(doc, m.group(1).strip(), seen)
            return m.group(0) if resolved is None else str(resolved)

        return _INTERPOLATION_RE.sub(substitute, value)
    if isinstance(value, dict):
        return {k: _resolve_value(doc, v, seen) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(doc, v, seen) for v in value]
    return value


def _lookup(doc: dict, dotted_path: str, seen: frozenset) -> Any:
    if dotted_path in seen:
        return None  # interpolation cycle - bail out rather than loop forever

    cursor: Any = doc
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return _resolve_value(doc, cursor, seen | {dotted_path})


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
