"""Extract pinned dependency versions relevant to GPU training.

Walks a directory (up to a fixed depth) looking for requirements.txt,
environment.yml, uv.lock, poetry.lock, or Pipfile.lock, and pulls out
whatever constraint each one puts on a small set of packages that matter
for the kind of misconfiguration traincheck cares about: torch, the
nvidia-nccl-cu*/nvidia-cuda* wheel families, deepspeed, transformers, apex,
accelerate, megatron-core.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_TARGET_FILENAMES = {
    "requirements.txt",
    "environment.yml",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
}

_EXACT_PACKAGES = {"torch", "deepspeed", "transformers", "apex", "accelerate", "megatron-core"}
_PREFIX_PACKAGES = ("nvidia-nccl-cu", "nvidia-cuda")

_REQ_LINE_RE = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*(.*)$")


def extract_lockfile(base_dir: str, max_depth: int = 3) -> dict:
    """Return {package_name: constraint_string} for every tracked package
    found across any lock/requirements file under `base_dir`.
    """
    constraints = {}
    for path in sorted(_find_lockfiles(Path(base_dir), max_depth)):
        constraints.update(_parse_file(path))
    return constraints


def _find_lockfiles(base_dir: Path, max_depth: int) -> list:
    found = []
    stack = [(base_dir, 0)]
    while stack:
        current, depth = stack.pop()
        if not current.is_dir():
            continue
        for entry in current.iterdir():
            if entry.is_file() and entry.name in _TARGET_FILENAMES:
                found.append(entry)
            elif entry.is_dir() and depth < max_depth:
                stack.append((entry, depth + 1))
    return found


def _parse_file(path: Path) -> dict:
    try:
        text = path.read_text()
    except OSError:
        return {}

    parser = {
        "requirements.txt": _parse_requirements,
        "environment.yml": _parse_environment_yml,
        "uv.lock": _parse_toml_lock,
        "poetry.lock": _parse_toml_lock,
        "Pipfile.lock": _parse_pipfile_lock,
    }.get(path.name)
    return parser(text) if parser else {}


def _is_tracked(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    if normalized in _EXACT_PACKAGES:
        return True
    return any(normalized.startswith(prefix) for prefix in _PREFIX_PACKAGES)


def _parse_requirement_line(line: str) -> Optional[tuple]:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    match = _REQ_LINE_RE.match(line)
    if not match:
        return None
    name, constraint = match.groups()
    return name, constraint.strip()


def _parse_requirements(text: str) -> dict:
    constraints = {}
    for raw_line in text.splitlines():
        parsed = _parse_requirement_line(raw_line)
        if parsed and _is_tracked(parsed[0]):
            constraints[parsed[0]] = parsed[1]
    return constraints


def parse_pip_list(pip_entries: list) -> dict:
    """Filter a bare list of pip requirement strings - e.g. a Ray
    runtime_env's "pip" key - down to the tracked packages, the same way
    a requirements.txt line would be.
    """
    constraints = {}
    for entry in pip_entries or []:
        parsed = _parse_requirement_line(entry)
        if parsed and _is_tracked(parsed[0]):
            constraints[parsed[0]] = parsed[1]
    return constraints


def _parse_environment_yml(text: str) -> dict:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(doc, dict):
        return {}

    constraints = {}
    for dep in doc.get("dependencies") or []:
        if isinstance(dep, str):
            parsed = _parse_requirement_line(dep)
            if parsed and _is_tracked(parsed[0]):
                constraints[parsed[0]] = parsed[1]
        elif isinstance(dep, dict):
            for pip_dep in dep.get("pip") or []:
                parsed = _parse_requirement_line(pip_dep)
                if parsed and _is_tracked(parsed[0]):
                    constraints[parsed[0]] = parsed[1]
    return constraints


def _parse_toml_lock(text: str) -> dict:
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}

    constraints = {}
    for package in doc.get("package") or []:
        name = package.get("name")
        version = package.get("version")
        if name and version and _is_tracked(name):
            constraints[name] = version
    return constraints


def _parse_pipfile_lock(text: str) -> dict:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return {}

    constraints = {}
    for section in ("default", "develop"):
        for name, spec in (doc.get(section) or {}).items():
            if _is_tracked(name) and isinstance(spec, dict):
                version = spec.get("version")
                if version:
                    constraints[name] = version
    return constraints
