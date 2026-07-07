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
    Rule(
        id="NCCL-2251-001",
        severity=Severity.ERROR,
        condition="nccl_version is not None and nccl_version == (2, 25, 1)",
        message=(
            "NCCL 2.25.1 hangs on ncclCommAbort during process-group teardown, unconditionally within "
            "this exact version. See github.com/pytorch/pytorch/issues/149153 - fix cherry-picked into "
            "the PyTorch 2.7 release branch."
        ),
        fix_suggestion="Upgrade to NCCL >= 2.26.2.",
    ),
    Rule(
        id="NCCL-2155-001",
        severity=Severity.WARN,
        condition=(
            "nccl_version is not None and nccl_version < (2, 15, 5) "
            "and gpu_type is not None and str(gpu_type).startswith('H100') "
            "and nccl_algo == 'Ring' "
            "and nodes is not None and gpus_per_node is not None "
            "and nodes * gpus_per_node == 2"
        ),
        message=(
            "NCCL versions before 2.15.5 hang on H100 GPUs using the Ring algorithm with the LL128 "
            "protocol on exactly 2 total GPUs. NVIDIA shipped a workaround (not a full root fix) in "
            "2.15.5 - see docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-15-5.html."
        ),
        fix_suggestion="Upgrade to NCCL >= 2.15.5, or avoid NCCL_ALGO=Ring on a 2-GPU H100 job.",
    ),
    Rule(
        id="CUDNN-LAYERNORM-001",
        severity=Severity.WARN,
        condition=(
            "cuda_version is not None and cuda_version[:2] >= (13, 1) "
            "and gpu_type is not None "
            "and (str(gpu_type).startswith('B200') or str(gpu_type).startswith('H200') "
            "or str(gpu_type).startswith('L40S') or str(gpu_type).startswith('A100'))"
        ),
        message=(
            "cuDNN's LayerNorm/RMSNorm kernels are slower on CUDA Toolkit 13.1+ than on CUDA 13.0 "
            "Update 2, on B200/H200/L40S/A100 GPUs - a performance regression, not a correctness bug. "
            "NVIDIA has a fix planned for a future release; see "
            "docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html."
        ),
        fix_suggestion=(
            "No fix yet - if throughput matters more than the newer toolkit, stay on CUDA <= 13.0 Update 2 for now."
        ),
    ),
    Rule(
        id="CUBLAS-HOPPER-MPS-001",
        severity=Severity.WARN,
        condition=(
            "cuda_version is not None and cuda_version[:2] in ((12, 2), (12, 3)) "
            "and gpu_type is not None and str(gpu_type).startswith('H100')"
        ),
        message=(
            "CUDA 12.2-12.3 has a known cuBLAS initialization failure on Hopper (H100) GPUs when "
            "running under MPS with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE below 100% - only applies if this "
            "job uses MPS, which traincheck cannot detect from the config alone. traincheck can only "
            "compare CUDA at major.minor precision, so this also fires on 12.3 Update 1+ (already fixed) "
            "since that's indistinguishable from bare 12.3 at this precision - check your patch/update "
            "level before acting. See docs.nvidia.com/cuda/archive/12.2.1/cuda-toolkit-release-notes "
            "(known issue) - fixed in cuBLAS 12.3 Update 1."
        ),
        fix_suggestion=(
            "Upgrade to CUDA >= 12.4.0 (cuBLAS 12.3 Update 1 or later), or don't use MPS with a reduced "
            "thread percentage."
        ),
    ),
    Rule(
        id="NCCL-GB200-001",
        severity=Severity.WARN,
        condition=(
            "nccl_version is not None and nccl_version < (2, 29, 2) "
            "and gpu_type is not None "
            "and (str(gpu_type).startswith('GB200') or str(gpu_type).startswith('GB300')) "
            "and nccl_net_gdr_level is not None and nccl_net_gdr_level == 0"
        ),
        message=(
            "NCCL before 2.29.2 hangs on GB200/GB300 with GDR disabled (NCCL_NET_GDR_LEVEL=0). Exact "
            "start of the broken range is unconfirmed (NVIDIA's 2.29.2 release notes name the fix but no "
            "earlier release note names this as a known issue) and this additionally requires a CX8 NIC, "
            "which traincheck cannot verify. See docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-29-2.html."
        ),
        fix_suggestion="Upgrade to NCCL >= 2.29.2, or re-enable GDR (NCCL_NET_GDR_LEVEL > 0) if this hang is hit.",
    ),
    Rule(
        id="CUBLAS-HOPPER-EPILOGUE-001",
        severity=Severity.WARN,
        condition=(
            "cuda_version is not None and cuda_version[:2] >= (12, 2) "
            "and gpu_type is not None and str(gpu_type).startswith('H100')"
        ),
        message=(
            "CUDA 12.2+ has a known silent-data-corruption bug for Hopper (H100) batched matmuls using "
            "the CUBLASLT_EPILOGUE_RELU_BIAS or CUBLASLT_EPILOGUE_GELU_BIAS epilogue with a non-zero bias "
            "batch stride - only applies if a custom kernel uses one of those specific epilogues, which "
            "traincheck cannot detect. No confirmed fix version as of the last release notes checked "
            "(12.8.0). See docs.nvidia.com/cuda/archive/12.2.1/cuda-toolkit-release-notes."
        ),
        fix_suggestion=(
            "If using these epilogues on Hopper, verify results independently; no confirmed fixed "
            "version to upgrade to yet."
        ),
    ),
    Rule(
        id="CUDA-HOPPER-MMASP-001",
        severity=Severity.WARN,
        condition=(
            "cuda_version is not None and cuda_version[:2] in ((12, 2), (12, 3)) "
            "and gpu_type is not None and str(gpu_type).startswith('H100')"
        ),
        message=(
            "CUDA 12.2-12.3 has a known intermittent silent-data-corruption bug on Hopper (H100) for "
            "custom kernels using the mma.sp PTX sparsity instruction directly - NVIDIA's own libraries "
            "don't use this instruction, so only custom kernels calling it are affected, which traincheck "
            "cannot detect. Confirmed present through CUDA 12.3.2 specifically; no confirmed fix version, "
            "and traincheck can only compare at major.minor precision so this also fires on any later "
            "12.3.x patch. See docs.nvidia.com/cuda/archive/12.3.2/cuda-toolkit-release-notes."
        ),
        fix_suggestion=(
            "If using mma.sp directly on Hopper in this CUDA range, verify results independently; no "
            "confirmed fixed version yet."
        ),
    ),
    Rule(
        id="CUDA12X-DRIVER-001",
        severity=Severity.ERROR,
        condition=(
            "cuda_version is not None and (12, 0) <= cuda_version[:2] < (13, 0) "
            "and driver_version is not None "
            "and (int(driver_version.split('.')[0]), int(driver_version.split('.')[1]), "
            "int(driver_version.split('.')[2])) < (525, 60, 13)"
        ),
        message=(
            "CUDA 12.x requires driver >= 525.60.13 (Linux) / >= 528.33 (Windows) for minor-version "
            "compatibility - see Table 2 in docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes. "
            "This check uses the Linux threshold; Windows hosts need the higher 528.33 floor. This is a "
            "live host fact - it only evaluates (and otherwise surfaces as needs-verification) when "
            "driver_version has actually been resolved, e.g. via --probe-host."
        ),
        fix_suggestion="Upgrade the host driver to >= 525.60.13 (Linux) / >= 528.33 (Windows).",
    ),
    Rule(
        id="ACCELERATE-FSDP2-001",
        severity=Severity.WARN,
        condition=(
            "mixed_precision is not None and mixed_precision != 'no' "
            "and parse_pinned_version(dependency_constraint(dependency_constraints, 'accelerate')) == (1, 13, 0)"
        ),
        message=(
            "accelerate==1.13.0's FSDP2 fp32-upcast for mixed precision is a silent no-op (a PR regression "
            "reverted in 1.14.0) - master weights silently stay in the lower-precision dtype instead of "
            "upcasting to fp32, with no error or crash. See github.com/huggingface/accelerate/issues/3979."
        ),
        fix_suggestion="Upgrade accelerate to >= 1.14.0.",
    ),
    Rule(
        id="DEEPSPEED-ZERO2-001",
        severity=Severity.WARN,
        condition=(
            "sharding is not None and sharding == 2 "
            "and parse_pinned_version(dependency_constraint(dependency_constraints, 'deepspeed')) is not None "
            "and (0, 18, 5) <= parse_pinned_version(dependency_constraint(dependency_constraints, 'deepspeed')) "
            "< (0, 18, 7)"
        ),
        message=(
            "deepspeed 0.18.5-0.18.6 has a confirmed ZeRO-2 training-throughput regression (maintainer "
            "reproduced, reporter confirmed the fix restores performance). See "
            "github.com/deepspeedai/DeepSpeed/issues/7885."
        ),
        fix_suggestion="Upgrade deepspeed to >= 0.18.7.",
    ),
    Rule(
        id="APEX-GROUPNORM-001",
        severity=Severity.WARN,
        condition=(
            "cuda_version is not None and (12, 4) <= cuda_version[:2] < (12, 8) "
            "and dependency_constraint(dependency_constraints, 'apex') is not None"
        ),
        message=(
            "apex's GroupNorm V2 extension fails to build on CUDA 12.4-12.7 with 'nvcc fatal: Unsupported "
            "gpu architecture compute_100' (setup.py requested sm_100/sm_120 unconditionally) - confirmed "
            "by an NVIDIA engineer, fixed via a merged PR not yet in a tagged apex release as of when this "
            "was mined. See github.com/NVIDIA/apex/issues/1919."
        ),
        fix_suggestion="Build apex from a commit at or after PR #1919's fix, or use CUDA >= 12.8 in the meantime.",
    ),
    Rule(
        id="MEGATRON-DCP-001",
        severity=Severity.WARN,
        condition=(
            "framework_version is not None and framework_version >= (2, 9) "
            "and dependency_constraint(dependency_constraints, 'megatron-core') is not None"
        ),
        message=(
            "torch >= 2.9 added a dataclasses.replace() call in its distributed-checkpoint code that drops "
            "any dynamically-added attribute, breaking Megatron-LM/mcore's custom 'mcore_data' attribute on "
            "checkpoint save/load - confirmed by tracing torch's own source (present v2.9.0-2.10.0, absent "
            "in v2.8.0) and by pytorch/pytorch#162948, which names Megatron-LM as the affected downstream. "
            "PyTorch's own fix PR was closed without merging, so this is still open with no fixed version. "
            "This assumes framework_version reflects torch specifically, not Megatron's own version - check "
            "which one your 'framework' config block actually names."
        ),
        fix_suggestion=(
            "No confirmed fix yet; pin torch < 2.9 for Megatron/mcore jobs using distributed "
            "checkpointing until this is resolved upstream."
        ),
    ),
]
