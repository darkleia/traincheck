"""Tests for mined-and-verified version-incompatibility rules (see
mining/README.md and mining/rules_verified.jsonl for how each one here was
sourced - nothing in rules/version_incompat.py may be authored from
memory).
"""

from traincheck.core import RuleEngine
from traincheck.ir import Field
from traincheck.rules.version_incompat import VERSION_INCOMPAT_RULES
from traincheck.validator import JobSpec


def _fired_ids(spec: JobSpec) -> set:
    engine = RuleEngine()
    for rule in VERSION_INCOMPAT_RULES:
        engine.register(rule)
    result = engine.check(vars(spec))
    return {v.rule.id for v in result.violations}


def _spec(nccl_version, gpu_type) -> JobSpec:
    spec = JobSpec()
    spec.nccl_version = Field(nccl_version, status="resolved", source="test")
    spec.gpu_type = Field(gpu_type, status="resolved", source="test")
    return spec


def test_nccl_h100_001_fires_in_the_known_broken_range():
    assert "NCCL-H100-001" in _fired_ids(_spec((2, 18, 1), "H100-SXM5-80GB"))
    assert "NCCL-H100-001" in _fired_ids(_spec((2, 18, 2), "H100"))


def test_nccl_h100_001_does_not_fire_once_fixed():
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 18, 3), "H100"))
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 19, 0), "H100"))


def test_nccl_h100_001_does_not_fire_before_the_broken_range():
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 18, 0), "H100"))


def test_nccl_h100_001_does_not_fire_on_other_gpu_types():
    assert "NCCL-H100-001" not in _fired_ids(_spec((2, 18, 1), "A100-SXM4-80GB"))


def test_nccl_h100_001_message_cites_its_source():
    engine = RuleEngine()
    for rule in VERSION_INCOMPAT_RULES:
        engine.register(rule)
    result = engine.check(vars(_spec((2, 18, 1), "H100")))

    violation = next(v for v in result.violations if v.rule.id == "NCCL-H100-001")
    assert "docs.nvidia.com" in violation.rule.message


def _base_spec(**overrides) -> JobSpec:
    spec = JobSpec()
    for name, value in overrides.items():
        setattr(spec, name, Field(value, status="resolved", source="test"))
    return spec


def test_nccl_2251_001_fires_only_on_the_exact_broken_patch():
    assert "NCCL-2251-001" in _fired_ids(_base_spec(nccl_version=(2, 25, 1)))
    assert "NCCL-2251-001" not in _fired_ids(_base_spec(nccl_version=(2, 25, 0)))
    assert "NCCL-2251-001" not in _fired_ids(_base_spec(nccl_version=(2, 26, 2)))


def test_nccl_2155_001_fires_only_for_h100_ring_exactly_2_gpus_before_the_fix():
    broken = _base_spec(
        nccl_version=(2, 14, 3), gpu_type="H100-SXM5-80GB", nccl_algo="Ring", nodes=1, gpus_per_node=2
    )
    assert "NCCL-2155-001" in _fired_ids(broken)

    fixed = _base_spec(nccl_version=(2, 15, 5), gpu_type="H100", nccl_algo="Ring", nodes=1, gpus_per_node=2)
    assert "NCCL-2155-001" not in _fired_ids(fixed)

    more_gpus = _base_spec(nccl_version=(2, 14, 3), gpu_type="H100", nccl_algo="Ring", nodes=1, gpus_per_node=8)
    assert "NCCL-2155-001" not in _fired_ids(more_gpus)

    not_ring = _base_spec(nccl_version=(2, 14, 3), gpu_type="H100", nccl_algo="Tree", nodes=1, gpus_per_node=2)
    assert "NCCL-2155-001" not in _fired_ids(not_ring)


def test_cudnn_layernorm_001_fires_on_cuda_131_plus_named_gpus():
    for gpu in ("B200", "H200-SXM", "L40S", "A100-SXM4-80GB"):
        assert "CUDNN-LAYERNORM-001" in _fired_ids(_base_spec(cuda_version=(13, 1), gpu_type=gpu))

    assert "CUDNN-LAYERNORM-001" not in _fired_ids(_base_spec(cuda_version=(13, 0), gpu_type="B200"))
    assert "CUDNN-LAYERNORM-001" not in _fired_ids(_base_spec(cuda_version=(13, 1), gpu_type="V100"))
    # a bare major.minor cuda_version (e.g. from a "module load cuda/13.1")
    # must compare correctly against the 2-tuple boundary
    assert "CUDNN-LAYERNORM-001" in _fired_ids(_base_spec(cuda_version=(13, 1), gpu_type="H200"))


def test_cublas_hopper_mps_001_fires_on_122_or_123_h100_regardless_of_tuple_length():
    # 3-tuple (as from an image's CUDA_VERSION env var) and 2-tuple (as from
    # "module load cuda/12.2") must both be treated as inside the range -
    # this is the exact case the [:2] normalization exists to protect.
    assert "CUBLAS-HOPPER-MPS-001" in _fired_ids(_base_spec(cuda_version=(12, 2, 128), gpu_type="H100"))
    assert "CUBLAS-HOPPER-MPS-001" in _fired_ids(_base_spec(cuda_version=(12, 2), gpu_type="H100"))
    assert "CUBLAS-HOPPER-MPS-001" in _fired_ids(_base_spec(cuda_version=(12, 3), gpu_type="H100"))

    assert "CUBLAS-HOPPER-MPS-001" not in _fired_ids(_base_spec(cuda_version=(12, 4), gpu_type="H100"))
    assert "CUBLAS-HOPPER-MPS-001" not in _fired_ids(_base_spec(cuda_version=(12, 2), gpu_type="A100"))


def test_nccl_gb200_001_fires_only_with_gdr_disabled_before_the_fix():
    broken = _base_spec(nccl_version=(2, 28, 3), gpu_type="GB200", nccl_net_gdr_level=0)
    assert "NCCL-GB200-001" in _fired_ids(broken)

    gdr_enabled = _base_spec(nccl_version=(2, 28, 3), gpu_type="GB200", nccl_net_gdr_level=5)
    assert "NCCL-GB200-001" not in _fired_ids(gdr_enabled)

    fixed = _base_spec(nccl_version=(2, 29, 2), gpu_type="GB300", nccl_net_gdr_level=0)
    assert "NCCL-GB200-001" not in _fired_ids(fixed)

    wrong_gpu = _base_spec(nccl_version=(2, 28, 3), gpu_type="H100", nccl_net_gdr_level=0)
    assert "NCCL-GB200-001" not in _fired_ids(wrong_gpu)


def test_cublas_hopper_epilogue_001_fires_from_122_onward_on_h100_only():
    assert "CUBLAS-HOPPER-EPILOGUE-001" in _fired_ids(_base_spec(cuda_version=(12, 2), gpu_type="H100"))
    assert "CUBLAS-HOPPER-EPILOGUE-001" in _fired_ids(_base_spec(cuda_version=(12, 8), gpu_type="H100"))
    assert "CUBLAS-HOPPER-EPILOGUE-001" not in _fired_ids(_base_spec(cuda_version=(12, 1), gpu_type="H100"))
    assert "CUBLAS-HOPPER-EPILOGUE-001" not in _fired_ids(_base_spec(cuda_version=(12, 8), gpu_type="A100"))


def test_cuda_hopper_mmasp_001_fires_on_122_or_123_h100_only():
    assert "CUDA-HOPPER-MMASP-001" in _fired_ids(_base_spec(cuda_version=(12, 2), gpu_type="H100"))
    assert "CUDA-HOPPER-MMASP-001" in _fired_ids(_base_spec(cuda_version=(12, 3, 2), gpu_type="H100"))
    assert "CUDA-HOPPER-MMASP-001" not in _fired_ids(_base_spec(cuda_version=(12, 4), gpu_type="H100"))
    assert "CUDA-HOPPER-MMASP-001" not in _fired_ids(_base_spec(cuda_version=(12, 2), gpu_type="A100"))


def test_cuda12x_driver_001_fires_when_driver_resolved_below_the_floor():
    spec = _base_spec(cuda_version=(12, 4))
    spec.driver_version = Field("520.61.05", status="resolved", source="test")
    assert "CUDA12X-DRIVER-001" in _fired_ids(spec)


def test_cuda12x_driver_001_does_not_fire_when_driver_meets_the_floor():
    spec = _base_spec(cuda_version=(12, 4))
    spec.driver_version = Field("535.129.03", status="resolved", source="test")
    assert "CUDA12X-DRIVER-001" not in _fired_ids(spec)


def test_accelerate_fsdp2_001_fires_only_on_the_exact_broken_pin():
    broken = _base_spec(mixed_precision="bf16")
    broken.dependency_constraints = Field({"accelerate": "==1.13.0"}, status="resolved", source="test")
    assert "ACCELERATE-FSDP2-001" in _fired_ids(broken)

    fixed = _base_spec(mixed_precision="bf16")
    fixed.dependency_constraints = Field({"accelerate": "==1.14.0"}, status="resolved", source="test")
    assert "ACCELERATE-FSDP2-001" not in _fired_ids(fixed)

    no_mp = _base_spec(mixed_precision="no")
    no_mp.dependency_constraints = Field({"accelerate": "==1.13.0"}, status="resolved", source="test")
    assert "ACCELERATE-FSDP2-001" not in _fired_ids(no_mp)

    # a loose range constraint can't be collapsed to one version - must not
    # guess that it lands on the broken pin
    ranged = _base_spec(mixed_precision="bf16")
    ranged.dependency_constraints = Field({"accelerate": ">=1.13.0,<1.14.0"}, status="resolved", source="test")
    assert "ACCELERATE-FSDP2-001" not in _fired_ids(ranged)


def test_deepspeed_zero2_001_fires_only_in_the_broken_range_at_zero_stage_2():
    broken = _base_spec(sharding=2)
    broken.dependency_constraints = Field({"deepspeed": "0.18.5"}, status="resolved", source="test")
    assert "DEEPSPEED-ZERO2-001" in _fired_ids(broken)

    fixed = _base_spec(sharding=2)
    fixed.dependency_constraints = Field({"deepspeed": "0.18.7"}, status="resolved", source="test")
    assert "DEEPSPEED-ZERO2-001" not in _fired_ids(fixed)

    wrong_stage = _base_spec(sharding=3)
    wrong_stage.dependency_constraints = Field({"deepspeed": "0.18.5"}, status="resolved", source="test")
    assert "DEEPSPEED-ZERO2-001" not in _fired_ids(wrong_stage)


def test_apex_groupnorm_001_fires_on_cuda_124_to_127_when_apex_is_pinned():
    broken = _base_spec(cuda_version=(12, 6))
    broken.dependency_constraints = Field({"apex": "==0.1"}, status="resolved", source="test")
    assert "APEX-GROUPNORM-001" in _fired_ids(broken)

    no_apex = _base_spec(cuda_version=(12, 6))
    no_apex.dependency_constraints = Field({"torch": "==2.3.0"}, status="resolved", source="test")
    assert "APEX-GROUPNORM-001" not in _fired_ids(no_apex)

    fixed_cuda = _base_spec(cuda_version=(12, 8))
    fixed_cuda.dependency_constraints = Field({"apex": "==0.1"}, status="resolved", source="test")
    assert "APEX-GROUPNORM-001" not in _fired_ids(fixed_cuda)


def test_megatron_dcp_001_fires_on_torch_29_plus_when_megatron_core_is_pinned():
    broken = _base_spec(framework_version=(2, 9, 0))
    broken.dependency_constraints = Field({"megatron-core": "==0.9.0"}, status="resolved", source="test")
    assert "MEGATRON-DCP-001" in _fired_ids(broken)

    # underscore spelling in the lockfile must still match
    broken_underscore = _base_spec(framework_version=(2, 9, 0))
    broken_underscore.dependency_constraints = Field({"megatron_core": "==0.9.0"}, status="resolved", source="test")
    assert "MEGATRON-DCP-001" in _fired_ids(broken_underscore)

    old_torch = _base_spec(framework_version=(2, 8, 0))
    old_torch.dependency_constraints = Field({"megatron-core": "==0.9.0"}, status="resolved", source="test")
    assert "MEGATRON-DCP-001" not in _fired_ids(old_torch)

    no_megatron = _base_spec(framework_version=(2, 9, 0))
    no_megatron.dependency_constraints = Field({"torch": "==2.9.0"}, status="resolved", source="test")
    assert "MEGATRON-DCP-001" not in _fired_ids(no_megatron)


def test_cuda12x_driver_001_routes_to_needs_verification_when_driver_unresolved():
    """The host-dependent candidate this rule was promoted from should never
    become a hard pass/fail without --probe-host - it must use the engine's
    existing unknown-status routing, not a new probing mechanism.
    """
    spec = _base_spec(cuda_version=(12, 4))
    spec.driver_version = Field(None, status="unknown", reason="host fact, not in any file")

    engine = RuleEngine()
    for rule in VERSION_INCOMPAT_RULES:
        engine.register(rule)
    result = engine.check(vars(spec))

    assert "CUDA12X-DRIVER-001" not in {v.rule.id for v in result.violations}
    assert "CUDA12X-DRIVER-001" in {nv.rule.id for nv in result.needs_verification}
