"""Map an LSF (bsub) job script onto a JobSpec.

Structurally identical to Slurm: a `#BSUB` directive block followed by a
plain shell body, so once the directives are read, the body is handed to
the same `apply_shell_body` every HPC scheduler adapter uses.

LSF directive values are sometimes quoted (`-R "span[ptile=8]"`, `-gpu
"num=8:mode=exclusive_process"`), so node/GPU counts need a bit more care
than a flat flag table:

- node count comes directly from `-nnodes` when a site has that resource
  connector enabled; otherwise (as in the far more common classic setup)
  it's `-n` (total tasks) divided by `-R`'s `span[ptile=N]` (tasks per
  node) - both are read, direct `-nnodes` taking precedence when present.
- GPU count comes from `-gpu`'s `num=N` sub-value (GPUs per host).
"""

import re
from typing import Optional

from traincheck.adapters.hpc_shell import apply_shell_body
from traincheck.ir import resolved_or_absent
from traincheck.utils import safe_int
from traincheck.validator import JobSpec

_BSUB_DIRECTIVE_RE = re.compile(r'^#BSUB\s+(-\S+)\s+(".*?"|\S+)')
_BSUB_LINE_RE = re.compile(r"^\s*#BSUB\b")
_PTILE_RE = re.compile(r"ptile=(\d+)")
_GPU_NUM_RE = re.compile(r"num=(\d+)")


def adapt_lsf(path: str, base_dir: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    directives = _parse_bsub_directives(text)

    nodes = safe_int(directives.get("-nnodes"))
    if nodes is None:
        nodes = _nodes_from_span(directives.get("-n"), directives.get("-R"))

    gpus_per_node = _gpus_from_flag(directives.get("-gpu"))

    spec = JobSpec()
    spec.nodes = resolved_or_absent(nodes, "lsf")
    spec.gpus_per_node = resolved_or_absent(gpus_per_node, "lsf")
    world_size = nodes * gpus_per_node if nodes is not None and gpus_per_node is not None else None
    spec.world_size = resolved_or_absent(world_size, "lsf")
    spec.walltime = resolved_or_absent(directives.get("-W"), "lsf")
    spec.partition = resolved_or_absent(directives.get("-q"), "lsf")

    body = _strip_bsub_lines(text)
    apply_shell_body(spec, body, base_dir)

    return spec


def _parse_bsub_directives(text: str) -> dict[str, str]:
    directives: dict[str, str] = {}
    for line in text.splitlines():
        match = _BSUB_DIRECTIVE_RE.match(line.strip())
        if not match:
            continue
        flag, value = match.groups()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1]
        directives[flag] = value
    return directives


def _nodes_from_span(ntasks: Optional[str], span: Optional[str]) -> Optional[int]:
    """`-n N` is the total task count, not a node count; `-R "span[ptile=M]"`
    says how many of those tasks land on each host, so N / M is nodes.
    """
    n = safe_int(ntasks)
    if n is None or not span:
        return None
    match = _PTILE_RE.search(span)
    if not match:
        return None
    ptile = safe_int(match.group(1))
    if not ptile:
        return None
    return n // ptile


def _gpus_from_flag(gpu_value: Optional[str]) -> Optional[int]:
    if not gpu_value:
        return None
    match = _GPU_NUM_RE.search(gpu_value)
    return safe_int(match.group(1)) if match else None


def _strip_bsub_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _BSUB_LINE_RE.match(line))
