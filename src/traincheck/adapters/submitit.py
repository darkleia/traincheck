"""Map a submitit AutoExecutor job script onto a JobSpec.

submitit.AutoExecutor abstracts over several possible backends (Slurm,
LSF, local, ...) and picks one at runtime based on what's available in
the environment - something a static read of the script can't know for
certain. What the script itself commits to is its
`executor.update_parameters(...)` call: generic kwargs (nodes,
gpus_per_node, timeout_min) apply regardless of backend, and slurm_*-
prefixed ones are a strong signal the author is targeting Slurm
specifically. We read those as the target backend when present; if
they're absent, AutoExecutor most likely resolved to a local (non-
scheduled) executor instead, and that's noted in meta rather than assumed.
"""

import ast
from typing import Any

from traincheck.ir import Field, resolved_or_absent
from traincheck.validator import JobSpec

_GENERIC_KWARG_FIELDS = {
    "nodes": "nodes",
    "gpus_per_node": "gpus_per_node",
    "timeout_min": "walltime",
}
_SLURM_KWARG_FIELDS = {
    "slurm_partition": "partition",
}

_LOCAL_BACKEND_REASON = (
    "no slurm_* kwargs in update_parameters; submitit's AutoExecutor "
    "likely resolved to a local (non-scheduled) executor"
)

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")


def adapt_submitit(path: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    spec = JobSpec()
    source = "submitit"

    kwargs = _find_update_parameters_kwargs(text)

    for kwarg_name, field_name in {**_GENERIC_KWARG_FIELDS, **_SLURM_KWARG_FIELDS}.items():
        setattr(spec, field_name, resolved_or_absent(kwargs.get(kwarg_name), source))

    has_slurm_kwargs = any(key.startswith("slurm_") for key in kwargs)
    spec.meta.stack = Field(
        value="submitit",
        status="resolved",
        source=source,
        confidence=1.0,
        reason="" if has_slurm_kwargs else _LOCAL_BACKEND_REASON,
    )

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec


def _find_update_parameters_kwargs(text: str) -> dict:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_update_parameters_call(node):
            return _literal_kwargs(node)
    return {}


def _is_update_parameters_call(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "update_parameters"


def _literal_kwargs(call: ast.Call) -> dict:
    kwargs = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue  # **extra unpacking - not statically readable
        value = _literal(kw.value)
        if value is not None:
            kwargs[kw.arg] = value
    return kwargs


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
