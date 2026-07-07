"""Version-incompatibility rules: component X version + component Y version
= known broken combination (e.g. "NCCL 2.18.0-2.18.2 hangs with CUDA 12.0-12.1").

Unlike `config_coherence.py`, nothing here may be authored from model
memory - an LLM asserting a version range from training data is exactly
the hallucination this module exists to prevent. Every rule below must
have reached this file by surviving the full mining pipeline:

    1. Extracted from a real, fetched page (a GitHub issue with a linked
       fix, a vendor release note, ...) - never generated from memory.
    2. Quarantined in ../../../../mining/candidates.jsonl as a candidate,
       never written here directly.
    3. Verified by a separate, adversarial re-grounding pass (re-fetch the
       source, confirm the symptom and both version sides actually appear
       on the page, try to falsify the claim, check the trigger is
       something the resolver can actually populate).
    4. Promoted here manually, carrying its source_url and symptom into
       the rule's message so every fired rule points back at the report
       that justified it.

See mining/README.md for the full methodology, and mining/rules_verified.jsonl
for the verified record each rule below was promoted from. The one
version-incompatibility claim that used to live in the old flat rules.py
(NCCL-RING-001, "NCCL Ring on A100 >32 nodes deadlocks below NCCL 2.21")
had no source_url and did not meet this bar - it has been quarantined into
mining/candidates.jsonl (status: "needs_remining") to be re-derived and
verified like any other mined candidate, rather than grandfathered in.
"""

from traincheck.core import Rule, Severity

VERSION_INCOMPAT_RULES: list[Rule] = [
    Rule(
        id="NCCL-H100-001",
        severity=Severity.WARN,
        condition=(
            "nccl_version is not None "
            "and (2, 18, 1) <= nccl_version < (2, 18, 3) "
            "and gpu_type is not None "
            "and str(gpu_type).startswith('H100')"
        ),
        message=(
            "NCCL 2.18.1-2.18.2 on H100 GPUs has a known AllReduce data-corruption risk when a NIC "
            "shares a PCI switch with only one GPU (this makes the LL128 protocol unsafe). Only applies "
            "if your topology matches - see docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-18-1.html "
            "(known issue) and rel_2-18-3.html (fix)."
        ),
        fix_suggestion=(
            "Upgrade to NCCL >= 2.18.3, or if you can't confirm your PCI topology avoids this, "
            "work around it with NCCL_PROTO=^LL128 or NCCL_IB_PCI_RELAXED_ORDERING=0."
        ),
    ),
]
