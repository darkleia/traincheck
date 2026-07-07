"""Extract launcher signals from a shell/sbatch script body.

Handles the parts of a job script that sit between the scheduler header
(e.g. #SBATCH directives, which are a separate concern) and the training
code: module loads, exported environment variables, the container image,
and the actual launch command (torchrun/accelerate/deepspeed/...).
"""

import re
from typing import Any, Callable, Optional

import bashlex
import bashlex.errors

_MODULE_LOAD_RE = re.compile(r"^\s*module\s+load\s+(.+)$")
_EXPORT_RE = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SOURCE_RE = re.compile(r"^\s*(?:source|\.)\s+\S+")

_LAUNCHER_KINDS = ("torchrun", "accelerate", "deepspeed", "mpirun", "horovodrun")

# `python -m torch.distributed.run` is torchrun under a different spelling -
# same flags, same elastic-agent semantics - so it maps to the same kind.
# `python -m torch.distributed.launch` is the older, pre-elastic-agent
# launcher; it's kept as its own kind so downstream can tell the two apart
# (e.g. it never gets the torchrun-only --max-restarts default).
_PYTHON_M_MODULE_KINDS = {
    "torch.distributed.run": "torchrun",
    "torch.distributed.launch": "torch.distributed.launch",
}

_CONTAINER_IMAGE_RE = re.compile(r"--container-image=(\S+)")
_IMAGE_FLAG_RE = re.compile(r"--image=(\S+)")
_SINGULARITY_RE = re.compile(r"singularity\s+(?:exec|run)\s+(?:-\S+(?:\s+\S+)?\s+)*(\S+)")
_DOCKER_RUN_RE = re.compile(r"docker\s+run\s+(?:-\S+(?:\s+\S+)?\s+)*(\S+)")
_IMAGE_PATTERNS = (_CONTAINER_IMAGE_RE, _IMAGE_FLAG_RE, _SINGULARITY_RE, _DOCKER_RUN_RE)

# Every value-taking launcher flag we understand, mapped to one canonical
# destination key. torchrun accepts both dash and underscore spellings for
# most of these, so every accepted spelling is listed - the scanner below
# is otherwise spelling-agnostic.
_VALUE_FLAGS = {
    "--nnodes": "nnodes",
    "--nproc-per-node": "nproc_per_node",
    "--nproc_per_node": "nproc_per_node",
    "--rdzv-backend": "rdzv_backend",
    "--rdzv_backend": "rdzv_backend",
    "--rdzv-endpoint": "rdzv_endpoint",
    "--rdzv_endpoint": "rdzv_endpoint",
    "--rdzv-id": "rdzv_id",
    "--rdzv_id": "rdzv_id",
    "--node-rank": "node_rank",
    "--node_rank": "node_rank",
    "--master-addr": "master_addr",
    "--master_addr": "master_addr",
    "--master-port": "master_port",
    "--master_port": "master_port",
    "--max-restarts": "max_restarts",
    "--max_restarts": "max_restarts",
    "--deepspeed": "deepspeed",
    "--config": "config",
    "--config-name": "config",
    "--config_name": "config",
}
# Boolean switches: presence alone is the signal, no value token follows.
_SWITCH_FLAGS = {
    "--standalone": "standalone",
    # Legacy torch.distributed.launch flag (deprecated even there); no
    # JobSpec field reads it yet, but it must still be recognized so it
    # isn't mistaken for a bare config override.
    "--use_env": "use_env",
}

_HOST_DEPENDENT_NPROC = {"gpu", "cpu", "xpu", "auto"}

_BARE_OVERRIDE_RE = re.compile(r"^[A-Za-z_][\w.]*=[^=\s]+$")
_VAR_REF_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


def extract_shell(script_text: str, base_dir: str, extra_env: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Pull launcher-relevant signals out of a shell/sbatch script body.

    `base_dir` is accepted for future use (resolving relative config paths
    against the script's own directory) but isn't needed by any signal
    extracted today.

    `extra_env` seeds env vars the script itself never exports but that are
    injected by whatever runs it (e.g. Slurm's own SLURM_GPUS_ON_NODE/
    SLURM_NTASKS) - a script export of the same name still overrides it,
    exactly as it would at runtime.
    """
    joined = _join_line_continuations(script_text)

    module_loads: list[str] = []
    env_vars: dict[str, str] = dict(extra_env or {})
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

        is_launcher_line = any(kind in line for kind in _LAUNCHER_KINDS) or any(
            module in line for module in _PYTHON_M_MODULE_KINDS
        )
        if launcher_line is None and is_launcher_line:
            launcher_line = line

    launcher, framework_config, config_path, config_overrides = _parse_launcher_line(launcher_line, env_vars)

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
    return parse_launcher_tokens(tokens, lambda value: _resolve_var(value, env_vars))


def parse_launcher_tokens(
    tokens: list[str], resolve_var: Callable[[Optional[str]], Optional[str]]
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str], list[str]]:
    """Work out everything torchrun/torch.distributed.launch/accelerate/...
    flags imply about the launcher, given an already-tokenized command.

    `tokens` can come from anywhere a launch command is spelled out as
    discrete argv entries - a shell line split by `_tokenize` below, or a
    Kubernetes container's `command`/`args` list, which is already a clean
    argv array with no shell-quoting to resolve.

    `resolve_var` resolves whatever variable-reference syntax the token
    source uses ($VAR/${VAR} for a shell script, $(VAR) for a Kubernetes
    container command) against whatever's known at parse time - returning
    None for anything injected later (by the shell's own runtime, or by a
    Kubernetes operator/kubelet) that can't be read statically.
    """
    if not tokens:
        return None, None, None, []

    kind = next((k for k in _LAUNCHER_KINDS if k in tokens), None)
    if kind is None:
        module = next((m for m in _PYTHON_M_MODULE_KINDS if m in tokens), None)
        kind = _PYTHON_M_MODULE_KINDS.get(module) if module else None
    raw, switches, config_overrides = _scan_tokens(tokens)

    def resolved(key: str) -> Optional[str]:
        return resolve_var(raw.get(key))

    nnodes_min, nnodes_max = _parse_nnodes(resolved("nnodes"))
    nproc_per_node, nproc_host_dependent = parse_nproc_value(resolved("nproc_per_node"))
    max_restarts_raw = resolved("max_restarts")
    standalone = "standalone" in switches
    rdzv_endpoint = resolved("rdzv_endpoint")

    launcher = {
        "kind": kind,
        # Convenience key for the common fixed-node-count case (min == max);
        # None for an elastic range, where nnodes_min/nnodes_max are the
        # only meaningful values.
        "nnodes": nnodes_min if nnodes_min == nnodes_max else None,
        "nnodes_min": nnodes_min,
        "nnodes_max": nnodes_max,
        "nproc_per_node": nproc_per_node,
        "nproc_per_node_host_dependent": nproc_host_dependent,
        "rdzv_backend": resolved("rdzv_backend"),
        "rdzv_endpoint": rdzv_endpoint,
        "rdzv_id": resolved("rdzv_id"),
        "node_rank": _as_int(resolved("node_rank")),
        "master_addr": resolved("master_addr"),
        "master_port": _as_int(resolved("master_port")),
        "max_restarts": _as_int(max_restarts_raw)
        if max_restarts_raw is not None
        else (0 if kind == "torchrun" else None),
        "max_restarts_is_default": max_restarts_raw is None and kind == "torchrun",
        "standalone": standalone,
        "standalone_conflict": standalone and rdzv_endpoint is not None,
    }
    framework_config = raw.get("deepspeed")
    config_path = resolved("config")
    return launcher, framework_config, config_path, config_overrides


def _scan_tokens(tokens: list[str]) -> tuple[dict[str, str], set[str], list[str]]:
    raw: dict[str, str] = {}
    switches: set[str] = set()
    config_overrides: list[str] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in _SWITCH_FLAGS:
            switches.add(_SWITCH_FLAGS[token])
            i += 1
            continue

        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            key = _VALUE_FLAGS.get(flag)
            if key:
                raw[key] = value
            i += 1
            continue

        key = _VALUE_FLAGS.get(token)
        if key and i + 1 < len(tokens):
            raw[key] = tokens[i + 1]
            i += 2
            continue

        if not token.startswith("-") and _BARE_OVERRIDE_RE.match(token):
            config_overrides.append(token)

        i += 1

    return raw, switches, config_overrides


def _parse_nnodes(value: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if value is None:
        return None, None
    if ":" in value:
        lo, _, hi = value.partition(":")
        return _as_int(lo), _as_int(hi)
    n = _as_int(value)
    return n, n


def parse_nproc_value(value: Optional[str]) -> tuple[Optional[int], bool]:
    """Parse a --nproc-per-node-shaped value: an integer, or one of the
    host-dependent tokens (gpu/cpu/xpu/auto) torchrun also accepts, which
    can only be resolved on the actual host - reused as-is for Kubernetes'
    equivalent `nprocPerNode` spec field, which allows the same tokens.
    """
    if value is None:
        return None, False
    if value.lower() in _HOST_DEPENDENT_NPROC:
        return None, True
    return _as_int(value), False


def _tokenize(line: str) -> list[str]:
    try:
        trees = bashlex.parse(line)
    except (bashlex.errors.ParsingError, NotImplementedError):
        # NotImplementedError: bashlex's grammar doesn't cover every real
        # shell construct (e.g. `$((...))` arithmetic expansion) - fall
        # back to a naive split rather than let a script feature bashlex
        # doesn't support crash the whole extraction.
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


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _resolve_var(value: Optional[str], env_vars: dict[str, str]) -> Optional[str]:
    """Resolve a `$VAR` reference against exports seen earlier in the same
    script. Returns None (rather than the raw `$VAR` text) when the
    variable comes from outside the script, since we can't know its value -
    and also when `$` appears inside a larger composite (e.g. a
    "$HOST:$PORT" endpoint), since substituting just one half would produce
    a value that looks real but isn't.
    """
    if value is None:
        return None
    match = _VAR_REF_RE.match(value)
    if match:
        return env_vars.get(match.group(1))
    return None if "$" in value else value
