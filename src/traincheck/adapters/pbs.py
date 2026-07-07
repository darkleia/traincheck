"""Map a PBS (Torque/PBS Pro) job script onto a JobSpec.

Structurally identical to Slurm: a `#PBS` directive block followed by a
plain shell body, so once the directives are read, the body is handed to
the same `apply_shell_body` every HPC scheduler adapter uses.

PBS's own resource request is a `-l` flag whose value is a colon-separated
`key=value` list (e.g. `-l select=8:ngpus=8:mpiprocs=8`), and `-l` can
appear more than once in the same header (one entry per resource type) -
every occurrence is merged into one flat resource dict before reading
node/GPU counts out of it. Modern PBS Pro spells node count "select" and
the classic Torque form spells it "nodes"; GPU count is "ngpus" or "gpu"
depending on the site's resource definitions - both spellings are read
either way.
"""

import re
from typing import Optional

from traincheck.adapters.hpc_shell import apply_shell_body
from traincheck.ir import resolved_or_absent
from traincheck.utils import safe_int
from traincheck.validator import JobSpec

_PBS_DIRECTIVE_RE = re.compile(r"^#PBS\s+(-\S+)(?:[=\s]+(\S+))?")
_PBS_LINE_RE = re.compile(r"^\s*#PBS\b")


def adapt_pbs(path: str, base_dir: str) -> JobSpec:
    with open(path) as f:
        text = f.read()

    directives = _parse_pbs_directives(text)
    resources = _parse_resource_lists(directives.get("-l") or [])

    spec = JobSpec()
    nodes = safe_int(resources.get("select") or resources.get("nodes"))
    gpus_per_node = safe_int(resources.get("ngpus") or resources.get("gpu"))

    spec.nodes = resolved_or_absent(nodes, "pbs")
    spec.gpus_per_node = resolved_or_absent(gpus_per_node, "pbs")
    world_size = nodes * gpus_per_node if nodes is not None and gpus_per_node is not None else None
    spec.world_size = resolved_or_absent(world_size, "pbs")
    spec.walltime = resolved_or_absent(resources.get("walltime"), "pbs")
    spec.partition = resolved_or_absent(_last(directives.get("-q")), "pbs")

    body = _strip_pbs_lines(text)
    apply_shell_body(spec, body, base_dir)

    return spec


def _parse_pbs_directives(text: str) -> dict[str, list]:
    """Every `#PBS -flag [value]` line, grouped by flag - `-l` legitimately
    repeats (one per resource type), so each flag maps to a list of every
    value seen for it, in order.
    """
    directives: dict[str, list] = {}
    for line in text.splitlines():
        match = _PBS_DIRECTIVE_RE.match(line.strip())
        if not match:
            continue
        flag, value = match.groups()
        if value is not None:
            directives.setdefault(flag, []).append(value)
    return directives


def _parse_resource_lists(values: list) -> dict[str, str]:
    """Each `-l` value is a colon-separated `key=value` list (e.g.
    "select=8:ngpus=8:mpiprocs=8"); every occurrence across the header is
    merged into one flat dict, last write wins.

    A colon is also PBS's own separator *within* a value (walltime is
    "HH:MM:SS"), so a fragment with no "=" of its own is a continuation of
    the previous key's value, not a new key.
    """
    resources: dict[str, str] = {}
    for value in values:
        key = None
        for part in value.split(":"):
            if "=" in part:
                key, _, val = part.partition("=")
                resources[key] = val
            elif key is not None:
                resources[key] += ":" + part
    return resources


def _last(values: Optional[list]) -> Optional[str]:
    return values[-1] if values else None


def _strip_pbs_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _PBS_LINE_RE.match(line))
