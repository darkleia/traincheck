"""Map an SGE (qsub) job script onto a JobSpec.

Structurally identical to Slurm: a `#$` directive block followed by a
plain shell body, so once the directives are read, the body is handed to
the same `apply_shell_body` every HPC scheduler adapter uses.

SGE has no first-class "node count" flag - `-pe <env> N` requests N total
slots from a parallel environment, and how those slots land on hosts is up
to that PE's own allocation rule (configured on the SGE master, not
visible in the job script). Given a per-node GPU count (`-l gpu=`),
though, N / that count is the same "total over per-node" trick already
used for Slurm's `--gpus` and LSF's `-n`/`span[ptile=]`, so it's used here
too, world_size taking the total slot count directly rather than a
node-count roundtrip.
"""

import re
from typing import Optional

from traincheck.adapters.hpc_shell import apply_shell_body
from traincheck.ir import resolved_or_absent
from traincheck.utils import safe_int
from traincheck.validator import JobSpec

_SGE_LINE_RE = re.compile(r"^\s*#\$")


def adapt_sge(path: str, base_dir: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    directives, slots = _parse_sge_directives(text)
    resources = _parse_l_resources(directives.get("-l") or [])

    gpus_per_node = safe_int(resources.get("gpu"))
    nodes = None
    if slots is not None and gpus_per_node:
        nodes = slots // gpus_per_node

    spec = JobSpec()
    spec.nodes = resolved_or_absent(nodes, "sge")
    spec.gpus_per_node = resolved_or_absent(gpus_per_node, "sge")
    world_size = slots if slots is not None else (nodes * gpus_per_node if nodes and gpus_per_node else None)
    spec.world_size = resolved_or_absent(world_size, "sge")
    spec.walltime = resolved_or_absent(resources.get("h_rt"), "sge")
    spec.partition = resolved_or_absent(_last(directives.get("-q")), "sge")

    body = _strip_sge_lines(text)
    apply_shell_body(spec, body, base_dir)

    return spec


def _parse_sge_directives(text: str) -> tuple[dict[str, list], Optional[int]]:
    """Every `#$ -flag [value ...]` line. `-pe <env> N` is its own special
    case (two values, only the slot count matters here); every other flag
    maps to a list of values seen for it, in order (mirroring `-l`, which
    can legitimately repeat).
    """
    directives: dict[str, list] = {}
    slots = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#$"):
            continue
        tokens = stripped[2:].split()
        if not tokens:
            continue

        flag, rest = tokens[0], tokens[1:]
        if flag == "-pe" and len(rest) >= 2:
            slots = safe_int(rest[1])
            continue
        if rest:
            directives.setdefault(flag, []).append(rest[0])

    return directives, slots


def _parse_l_resources(values: list) -> dict[str, str]:
    resources: dict[str, str] = {}
    for value in values:
        key, sep, val = value.partition("=")
        if sep:
            resources[key] = val
    return resources


def _last(values: Optional[list]) -> Optional[str]:
    return values[-1] if values else None


def _strip_sge_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _SGE_LINE_RE.match(line))
