"""Combine rule-triggered and host-fact needs-verification into one display-ready list.

`Result.needs_verification` (from the rule engine) only covers rules that
were actually blocked by an unresolved field. `JobSpec.meta.unresolved`
covers facts - like host driver/kernel/OFED/peermem state - that are
unknown regardless of whether any rule currently reads them. Both matter
to a human deciding whether it's safe to submit a job, so the CLI shows
them together.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional

from traincheck.core import Result
from traincheck.validator import JobSpec

HOST_ENV_CHECKS = {
    "driver_version": (
        "NVIDIA driver version",
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
    ),
    "kernel_version": ("running kernel version", "uname -r"),
    "ofed_version": ("OFED (InfiniBand driver stack) installed", "ofed_info -s"),
    "peermem_loaded": ("nvidia-peermem loaded", "lsmod | grep peermem"),
}


@dataclass
class VerificationItem:
    field_name: str
    reason: str
    rule_id: Optional[str] = None

    @property
    def check_command(self) -> Optional[str]:
        check = HOST_ENV_CHECKS.get(self.field_name)
        return check[1] if check else None

    @property
    def display(self) -> str:
        check = HOST_ENV_CHECKS.get(self.field_name)
        if check:
            description, command = check
            return f"verify {description}: {command}"
        return f"verify {self.field_name}: {self.reason}"


def collect_needs_verification(spec: JobSpec, result: Result) -> list:
    items = [
        VerificationItem(field_name=nv.field_name, reason=nv.reason, rule_id=nv.rule.id)
        for nv in result.needs_verification
    ]

    names_by_id = {
        id(getattr(spec, f.name)): f.name for f in dataclasses.fields(spec) if f.name != "meta"
    }
    if spec.meta.stack is not None:
        names_by_id[id(spec.meta.stack)] = "stack"

    seen = {item.field_name for item in items}
    for unresolved_field in spec.meta.unresolved:
        name = names_by_id.get(id(unresolved_field), "unknown_field")
        if name in seen:
            continue
        items.append(VerificationItem(field_name=name, reason=unresolved_field.reason))
        seen.add(name)

    return items
