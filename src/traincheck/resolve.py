"""Dispatch a config/script file to the right adapter based on its detected stack."""

import os

import yaml

from traincheck.adapters.slurm import adapt_slurm
from traincheck.detect import Stack, detect_stack
from traincheck.validator import JobSpec, parse_config


class UnsupportedStackError(Exception):
    """Raised when a file's stack can't be routed to any adapter yet."""


def resolve(path: str) -> JobSpec:
    stack = detect_stack(path)

    if stack == Stack.NATIVE:
        with open(path) as f:
            config = yaml.safe_load(f)
        return parse_config(config)

    if stack == Stack.SLURM:
        base_dir = os.path.dirname(os.path.abspath(path)) or "."
        return adapt_slurm(path, base_dir=base_dir)

    raise UnsupportedStackError(
        f"traincheck doesn't support this stack yet (detected: {stack.value}): {path}"
    )
