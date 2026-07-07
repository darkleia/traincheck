"""Map a bare Accelerate `default_config.yaml` onto a JobSpec.

Used when the config file itself is what traincheck was pointed at, as
opposed to a shell script that runs `accelerate launch --config_file ...`
- that case is handled by `extractors/accelerate.py`'s
`apply_accelerate_launch` instead (called from the HPC/bare-metal
adapters), which lets the launch line's own flags override the same file.
Both paths share the same underlying config-to-JobSpec mapping.
"""

from traincheck.extractors.accelerate import (
    apply_accelerate_config,
    extract_accelerate_config,
    route_embedded_frameworks,
)
from traincheck.ir import Field
from traincheck.validator import JobSpec

_HOST_ENV_REASON = "host fact, not in any file"
_HOST_ENV_FIELDS = ("driver_version", "kernel_version", "ofed_version", "peermem_loaded")


def adapt_accelerate(path: str, base_dir: str) -> JobSpec:
    """`base_dir` is accepted for signature parity with every other
    adapter but isn't needed - a default_config.yaml has no relative file
    references of its own to resolve.
    """
    fields = extract_accelerate_config(path)
    source = "accelerate"

    spec = JobSpec()
    apply_accelerate_config(spec, fields, source)
    route_embedded_frameworks(spec, fields, source)

    for name in _HOST_ENV_FIELDS:
        host_field = Field(value=None, status="unknown", reason=_HOST_ENV_REASON)
        setattr(spec, name, host_field)
        spec.meta.unresolved.append(host_field)

    return spec
