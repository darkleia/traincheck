"""Dispatch a config/script file to the right adapter based on its detected stack."""

import os

import yaml

from traincheck.adapters.accelerate import adapt_accelerate
from traincheck.adapters.bare import adapt_bare
from traincheck.adapters.k8s import adapt_k8s
from traincheck.adapters.lsf import adapt_lsf
from traincheck.adapters.pbs import adapt_pbs
from traincheck.adapters.ray import adapt_ray
from traincheck.adapters.sge import adapt_sge
from traincheck.adapters.skypilot import adapt_skypilot
from traincheck.adapters.slurm import adapt_slurm
from traincheck.adapters.submitit import adapt_submitit
from traincheck.adapters.torchx import adapt_torchx
from traincheck.detect import Stack, detect_stack
from traincheck.ir import Field
from traincheck.validator import JobSpec, parse_config

# Adapters that take (path, base_dir). submitit is handled separately since
# it needs neither - its script is fully self-contained.
_BASE_DIR_ADAPTERS = {
    Stack.SLURM: adapt_slurm,
    Stack.PBS: adapt_pbs,
    Stack.LSF: adapt_lsf,
    Stack.SGE: adapt_sge,
    Stack.K8S_CRD: adapt_k8s,
    Stack.SKYPILOT: adapt_skypilot,
    Stack.ACCELERATE: adapt_accelerate,
    Stack.RAY: adapt_ray,
    Stack.BARE: adapt_bare,
    Stack.TORCHX: adapt_torchx,
}


class UnsupportedStackError(Exception):
    """Raised when a file's stack can't be routed to any adapter yet."""


def resolve(path: str) -> JobSpec:
    stack = detect_stack(path)

    if stack == Stack.NATIVE:
        with open(path) as f:
            config = yaml.safe_load(f)
        spec = parse_config(config)
        _default_stack(spec, stack)
        return spec

    if stack == Stack.SUBMITIT:
        spec = adapt_submitit(path)
        _default_stack(spec, stack)
        return spec

    adapter = _BASE_DIR_ADAPTERS.get(stack)
    if adapter is not None:
        base_dir = os.path.dirname(os.path.abspath(path)) or "."
        spec = adapter(path, base_dir=base_dir)
        _default_stack(spec, stack)
        return spec

    raise UnsupportedStackError(f"traincheck doesn't support this stack yet (detected: {stack.value}): {path}")


def _default_stack(spec: JobSpec, stack: Stack) -> None:
    """Fill meta.stack with the detected stack name, unless the adapter
    already recorded something more specific (torchx reports the backend
    scheduler it delegated to, not "torchx" itself; submitit reports
    "submitit" either way, so there's nothing to override there either).
    """
    if spec.meta.stack is None:
        spec.meta.stack = Field(value=stack.value, status="resolved", source="resolve", confidence=1.0)
