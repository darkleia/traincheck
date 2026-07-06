"""Extract launcher signals from a shell/sbatch script body.

Handles the parts of a job script that sit between the scheduler header
(e.g. #SBATCH directives, which are a separate concern) and the training
code: module loads, exported environment variables, the container image,
and the actual launch command (torchrun/accelerate/deepspeed/...).
"""

import re
from typing import Any, Optional

import bashlex

_MODULE_LOAD_RE = re.compile(r"^\s*module\s+load\s+(.+)$")
_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SOURCE_RE = re.compile(r"^\s*(?:source|\.)\s+\S+")

_LAUNCHER_KINDS = ("torchrun", "accelerate", "deepspeed", "mpirun", "horovodrun")

_CONTAINER_IMAGE_RE = re.compile(r"--container-image=(\S+)")
_IMAGE_FLAG_RE = re.compile(r"--image=(\S+)")
_SINGULARITY_RE = re.compile(r"singularity\s+(?:exec|run)\s+(?:-\S+(?:\s+\S+)?\s+)*(\S+)")
_DOCKER_RUN_RE = re.compile(r"docker\s+run\s+(?:-\S+(?:\s+\S+)?\s+)*(\S+)")
_IMAGE_PATTERNS = (_CONTAINER_IMAGE_RE, _IMAGE_FLAG_RE, _SINGULARITY_RE, _DOCKER_RUN_RE)

_NNODES_FLAGS = ("--nnodes",)
_NPROC_FLAGS = ("--nproc-per-node", "--nproc_per_node")
_RDZV_BACKEND_FLAGS = ("--rdzv-backend", "--rdzv_backend")
_DEEPSPEED_FLAGS = ("--deepspeed",)
_CONFIG_FLAGS = ("--config", "--config-name", "--config_name")

_BARE_OVERRIDE_RE = re.compile(r"^[A-Za-z_][\w.]*=[^=\s]+$")
_VAR_REF_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


def extract_shell(script_text: str, base_dir: str) -> dict[str, Any]:
    """Pull launcher-relevant signals out of a shell/sbatch script body.

    `base_dir` is accepted for future use (resolving relative config paths
    against the script's own directory) but isn't needed by any signal
    extracted today.
    """
    joined = _join_line_continuations(script_text)

    module_loads: list[str] = []
    env_vars: dict[str, str] = {}
    image_ref: Optional[str] = None
    launcher_line: Optional[str] = None

    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        module_match = _MODULE_LOAD_RE.match(line)
        if module_match:
            module_loads.extend(module_match.group(1).split())
            continue

        export_match = _EXPORT_RE.match(line)
        if export_match:
            name, value = export_match.groups()
            env_vars[name] = _strip_quotes(value.strip())
            continue

        if _SOURCE_RE.match(line):
            continue

        if image_ref is None:
            image_ref = _find_image_ref(line)

        if launcher_line is None and any(kind in line for kind in _LAUNCHER_KINDS):
            launcher_line = line

    launcher, framework_config, config_path, config_overrides = _parse_launcher_line(
        launcher_line, env_vars
    )

    return {
        "module_loads": module_loads,
        "env_vars": env_vars,
        "image_ref": image_ref,
        "launcher": launcher,
        "framework_config": framework_config,
        "config_path": config_path,
        "config_overrides": config_overrides,
    }


def _join_line_continuations(text: str) -> str:
    return re.sub(r"\\\s*\n", " ", text)


def _find_image_ref(line: str) -> Optional[str]:
    for pattern in _IMAGE_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def _parse_launcher_line(
    launcher_line: Optional[str], env_vars: dict[str, str]
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str], list[str]]:
    if not launcher_line:
        return None, None, None, []

    tokens = _tokenize(launcher_line)
    kind = next((k for k in _LAUNCHER_KINDS if k in tokens), None)

    nnodes_raw: Optional[str] = None
    nproc_per_node_raw: Optional[str] = None
    rdzv_backend_raw: Optional[str] = None
    framework_config: Optional[str] = None
    config_path_raw: Optional[str] = None
    config_overrides: list[str] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag in _NNODES_FLAGS:
                nnodes_raw = value
            elif flag in _NPROC_FLAGS:
                nproc_per_node_raw = value
            elif flag in _RDZV_BACKEND_FLAGS:
                rdzv_backend_raw = value
            elif flag in _DEEPSPEED_FLAGS:
                framework_config = value
            elif flag in _CONFIG_FLAGS:
                config_path_raw = value
            i += 1
            continue

        has_next = i + 1 < len(tokens)
        if token in _NNODES_FLAGS and has_next:
            nnodes_raw = tokens[i + 1]
            i += 2
            continue
        if token in _NPROC_FLAGS and has_next:
            nproc_per_node_raw = tokens[i + 1]
            i += 2
            continue
        if token in _RDZV_BACKEND_FLAGS and has_next:
            rdzv_backend_raw = tokens[i + 1]
            i += 2
            continue
        if token in _DEEPSPEED_FLAGS and has_next:
            framework_config = tokens[i + 1]
            i += 2
            continue
        if token in _CONFIG_FLAGS and has_next:
            config_path_raw = tokens[i + 1]
            i += 2
            continue

        if not token.startswith("-") and _BARE_OVERRIDE_RE.match(token):
            config_overrides.append(token)

        i += 1

    nnodes_value = _resolve_var(nnodes_raw, env_vars)
    nproc_per_node_value = _resolve_var(nproc_per_node_raw, env_vars)
    launcher = {
        "kind": kind,
        "nnodes": _as_int(nnodes_value) if nnodes_value is not None else None,
        "nproc_per_node": _as_int(nproc_per_node_value) if nproc_per_node_value is not None else None,
        "rdzv_backend": _resolve_var(rdzv_backend_raw, env_vars),
    }
    config_path = _resolve_var(config_path_raw, env_vars)
    return launcher, framework_config, config_path, config_overrides


def _tokenize(line: str) -> list[str]:
    try:
        trees = bashlex.parse(line)
    except bashlex.errors.ParsingError:
        return line.split()

    words: list[str] = []
    for tree in trees:
        _collect_words(tree, words)
    return words


def _collect_words(node: Any, words: list[str]) -> None:
    if getattr(node, "kind", None) == "word":
        words.append(_strip_quotes(node.word))
        return
    for part in getattr(node, "parts", None) or []:
        _collect_words(part, words)


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _as_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


def _resolve_var(value: Optional[str], env_vars: dict[str, str]) -> Optional[str]:
    """Resolve a `$VAR` reference against exports seen earlier in the same
    script. Returns None (rather than the raw `$VAR` text) when the
    variable comes from outside the script, since we can't know its value.
    """
    if value is None:
        return None
    match = _VAR_REF_RE.match(value)
    if not match:
        return value
    return env_vars.get(match.group(1))
