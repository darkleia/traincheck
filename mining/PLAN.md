# Mining roadmap

A living checklist, ordered by expected yield. See `README.md` for the
methodology/schema/prompts this executes. Update the checkboxes and the
"done" notes as cells get worked - this file tracks progress across
sessions; `README.md` stays a static reference.

## Prompt B pass (verification) - run against all 8 `candidate` entries

All 8 survived adversarial re-grounding and moved to `status: verified` in
`mining/rules_verified.jsonl`: `nccl-2292-gb200-cx8-gdr-disabled-hang`,
`cublas-hopper-mps-init-fail`, `cublas-hopper-relu-gelu-bias-corruption`,
`cuda-hopper-mma-sp-sparsity-corruption`, `cuda12x-min-driver-52560`,
`nccl-2251-comm-abort-hang`, `nccl-2155-h100-ring-2gpu-hang`,
`cudnn-cuda131-layernorm-rmsnorm-perf-regression`. One real citation error
was caught and corrected in the process: `cublas-hopper-relu-gelu-bias-corruption`
had wrongly claimed to still be present in CUDA 12.8.0's release notes - the
archive pages are cumulative changelogs that reprint old releases' own
historical Known Issues verbatim, so a text match there isn't evidence of
current status. Re-checked against the raw page and corrected; the
underlying rule content didn't change (no upper bound was ever assertable
either way), only the citation trail. See each entry's notes in
`candidates.jsonl` for the specific re-verification done.

## Promotion pass - all 8 verified candidates now in `rules/version_incompat.py`

All 8 promoted, each with a dedicated test in `tests/test_version_incompat_rules.py`:
- `NCCL-2251-001` (ERROR) - from `nccl-2251-comm-abort-hang`. Unconditional
  hang within one exact NCCL version, no untracked precondition.
- `NCCL-2155-001` (WARN) - from `nccl-2155-h100-ring-2gpu-hang`. Fully
  expressible (nccl_version + gpu_type + nccl_algo + nodes*gpus_per_node).
- `CUDNN-LAYERNORM-001` (WARN) - from `cudnn-cuda131-layernorm-rmsnorm-perf-regression`.
  Fully expressible, no field gap, performance caveat stated in the message.
- `CUBLAS-HOPPER-MPS-001` (WARN) - from `cublas-hopper-mps-init-fail`. "Only
  if using MPS" caveat in the message, same necessary-but-not-sufficient
  pattern as `NCCL-H100-001`.
- `NCCL-GB200-001`, `CUBLAS-HOPPER-EPILOGUE-001`, `CUDA-HOPPER-MMASP-001`
  (all WARN) - from `nccl-2292-gb200-cx8-gdr-disabled-hang`,
  `cublas-hopper-relu-gelu-bias-corruption`, `cuda-hopper-mma-sp-sparsity-corruption`.
  Each message states its own unresolved caveat (unconfirmed range start,
  unconfirmed fix version, or an untrackable NIC/PTX/epilogue precondition).
- `CUDA12X-DRIVER-001` (ERROR when it fires) - from `cuda12x-min-driver-52560`.
  Converted to a needs-verification advisory using the *existing*
  unknown-status routing already in `RuleEngine.check()` (the same mechanism
  `NCCL-GDR-001` relies on) - no new probing code was written. `driver_version`
  defaults to `status="unknown"` in every adapter, so this rule only
  evaluates for real once `--probe-host` resolves it; until then it
  surfaces in `needs_verification`, never as a hard pass/fail.

**Real bug caught while writing these rules, fixed before promotion:**
`cuda_version` arrives as a 2-tuple from `module load cuda/X.Y` paths but a
3-tuple from image-env extraction (`CUDA_VERSION=12.2.128`) - Python treats
a shorter tuple as "less than" any longer tuple sharing its prefix, so a
chained `(12,2,0) <= cuda_version < (12,4,0)`-style condition silently
excludes a bare `(12,2)` value it should include. All new cuda_version
comparisons use `cuda_version[:2]` to normalize to major.minor precision
instead. Also fixed a separate, pre-existing bug while here:
`adapters/hpc_shell.py` parsed `nccl_version` from `module load` lines but
never applied `parse_version()` to `cuda_version` from the same code path -
it stayed a raw string, which would have silently broken every one of these
new rules for SLURM/PBS/LSF/SGE users specifically. See
`tests/test_version_incompat_rules.py`'s tuple-length-specific assertions
(e.g. `test_cublas_hopper_mps_001_fires_on_122_or_123_h100_regardless_of_tuple_length`)
for the regression coverage.

**Gate self-test:** `tests/test_mining_gate.py` now proves
`mining/validate_candidates.py` actually rejects bad input (missing
source_url, invalid status/confidence, oversized symptom, missing required
keys, unjustified high confidence) rather than just always printing "pass" -
run before trusting it to scale to more cells.

## Why this order

A verified candidate is only useful if it can become an active rule, which
needs both sides of the claim to map onto a JobSpec field traincheck
already populates (discipline #5). Mining a pair where one side isn't
tracked yet (e.g. an `accelerate` or `deepspeed` *package* version - as
opposed to the config *content* they read, which already is) still
produces a real, verified finding, but it lands in `candidates.jsonl` as
"verified but not yet expressible" rather than becoming an active rule.
Tier 1 below skips that outcome entirely; Tier 2 hits it by design, so its
value is scoped to also naming the exact JobSpec field that would need to
be added.

## Tier 1 - fully expressible today (mine first)

Both sides of every pair here are real, already-populated JobSpec fields
(`nccl_version`, `cuda_version`, `gpu_type`, `comm_env[...]`, `sharding`,
`tensor_parallel`/`pipeline_parallel`/..., `mixed_precision`). A verified
finding in this tier can become an active ERROR/WARN rule immediately.

- [x] **nccl × cuda** - NVIDIA NCCL release notes. Found & promoted
      NCCL-H100-001 (2.18.1-2.18.3/H100/LL128 corruption). Scanned 2.18.1,
      2.18.3, 2.19.3, 2.21.5, 2.26.2, 2.27.5, 2.28.3, 2.28.7, 2.28.9,
      2.29.2, 2.30.3 - also found a GB200/CX8/GDR-disabled hang
      (`nccl-2292-gb200-cx8-gdr-disabled-hang`, candidate, medium
      confidence: fix confirmed in 2.29.2 but the broken-range start isn't
      independently confirmed). Remaining unchecked: 2.14-2.17, 2.20,
      2.22-2.25 - lower priority (older, less likely to matter for current
      H100/B200/GB200-era jobs) but not exhausted.
- [x] **nccl × cuda** - `NVIDIA/nccl` GitHub issues, pages 1-2 of
      `is:issue is:closed "fixed in" (regression OR incompatible OR version)`.
      Rejected 2 more low-quality threads (#1051 Lightning hang - no
      confirmed fix; #1273 "message trutruncated" 2.20.5 vs 2.18.5 -
      closed not_planned, maintainer's fix guess was later walked back).
      Found one real, patch-confirmed bug not in the release notes
      (`nccl-223-profiler-memleak`, 2.23.x profiler memory leak) - rejected
      for promotion only because NCCL profiler-plugin usage isn't
      something traincheck can detect. Continue paginating past page 2.
- [x] **nccl × gpu_type** - effectively covered by the same release-notes
      scan above (H100/LL128, GB200/CX8 hang, Blackwell/sm80 mentions in
      2.28.3/2.29.2 were checked and didn't yield a clean version-bounded
      claim). Revisit only if doing a fresh, more exhaustive release-notes
      pass later.
- [x] **cuda × gpu_type** - CUDA Toolkit release notes (12.2.1, 12.3.2,
      12.4.0, 12.6.0, 12.8.0 checked). Found 3: cuBLAS/Hopper MPS init
      failure (`cublas-hopper-mps-init-fail`, high confidence, fixed in
      "cuBLAS 12.3 Update 1"/confirmed by 12.4.0's own resolved-issues
      list), a Hopper batched-matmul silent-corruption bug that's STILL
      OPEN as of 12.8.0 (`cublas-hopper-relu-gelu-bias-corruption`, medium
      confidence - no fix found), and an mma.sp PTX sparsity silent
      corruption bug present in 12.3.2 but no confirmed fix version
      (`cuda-hopper-mma-sp-sparsity-corruption`, medium). All 3 need a
      "does the trigger fully apply" caveat in the message (MPS usage /
      specific cuBLASLt epilogue / raw mma.sp PTX use aren't tracked) but
      are expressible on cuda_version + gpu_type alone as the necessary
      (not sufficient) condition - same pattern as NCCL-H100-001.
- [x] **nccl × comm_env vars** - re-mined the quarantined NCCL-RING-001
      claim specifically (Ring algo + A100 + >32 nodes + fixed at 2.21).
      Found nothing supporting it after real searches; stays
      `needs_remining` (not `rejected` - no specific refuting source, just
      an empty search) per the `rejected` vs `needs_remining` distinction
      now documented in README.md. Broader nccl×comm_env search (beyond
      re-deriving this one specific claim) not yet done.
- [x] **cuda × driver** - CUDA 12.x's own minimum-driver-version
      compatibility table (`cuda12x-min-driver-52560`, high confidence,
      `host_dependent: true` since driver_version is only known live).
      **nccl × driver** not yet done separately.
- [x] **pytorch × nccl (framework_version + nccl_version)** -
      `pytorch/pytorch` GitHub issues (`label:"module: nccl"`, one page).
      Found the cleanest candidate of the whole session:
      `nccl-2251-comm-abort-hang` (NCCL 2.25.1 ncclCommAbort hang, fixed in
      2.26.2, cherry-picked into PyTorch 2.7 - reporter self-identified the
      fix version, maintainer confirmed and closed). High confidence,
      **expressible on nccl_version alone**, no untracked precondition -
      this is the strongest promotion candidate after NCCL-H100-001.
      Rejected one inconclusive driver-vs-NCCL-version thread (#150852 -
      two unreconciled causal theories, no final confirmation). Not
      exhausted - only checked one page of one query.
- [x] **nccl × cuda / gpu_type** - filled in one more release-notes gap:
      2.15.5's Fixed Issues names `nccl-2155-h100-ring-2gpu-hang` (H100 +
      Ring algo + LL128 + exactly 2 GPUs hang, workaround shipped in 2.15.5).
      Fully expressible on nccl_version + gpu_type + nccl_algo +
      nodes*gpus_per_node - no untracked precondition, unlike the other H100
      candidates. Checked 2.14.3, 2.16.2, 2.17.1, 2.20.5, 2.22.3, 2.24.3,
      2.25.1 too - routine fixes only, nothing else version-bounded.
      NVIDIA/nccl GitHub search page 3 (`is:closed "fixed in" ...`) checked -
      all 2016-2019 legacy issues, zero new candidates, stopping here per
      the stop condition.
- [ ] **cuda × pytorch (framework_version)** - PyTorch's own
      release/compatibility matrix (which torch wheel pins which CUDA) -
      still not done as its own dedicated cell (distinct from the issues
      search above); deprioritized this pass in favor of the previously
      untouched Tier 2/3 cells below.

## Tier 2 - real bugs exist, but need one new field first

Mining these still produces legitimate, citable findings - they just land
in `candidates.jsonl` as verified-but-`expressible: false` until the named
field exists. Worth doing for the findings themselves, but expect no
immediately-promotable rules.

- [x] **deepspeed × pytorch** - `deepspeedai/DeepSpeed` GitHub issues
      (started; found `deepspeed-0192-zero3-peft-lora-dtype`, rejected -
      not expressible). Needs: a DeepSpeed config's own `bf16.enabled` flag
      (currently only `zero_stage`/offload/tp/pp/batch keys are read) and/or
      PEFT/LoRA usage detection - neither exists.
- [x] **accelerate × fsdp** - `huggingface/accelerate` GitHub issues
      (started; found `accelerate-1130-fsdp2-bf16-upcast-noop`, rejected -
      not expressible). Needs: an `accelerate` **package** version field
      (distinct from the `default_config.yaml` content already parsed) -
      e.g. read from a `requirements.txt`/lockfile pin near the config, the
      way `dependency_constraints` already works for Ray.
- [x] **deepspeed × cuda** - `deepspeedai/DeepSpeed` GitHub issues (found
      `deepspeed-0185-zero2-slowdown`, a maintainer+reporter-confirmed ZeRO-2
      perf regression from 0.18.5→0.18.7, rejected - not expressible). Same
      "deepspeed package version not tracked" gap as
      `deepspeed-0192-zero3-peft-lora-dtype` above - now the third finding
      blocked specifically by this one missing field.
- [x] **accelerate × deepspeed** - `huggingface/accelerate` GitHub issues
      (2 queries, ~50 issues/PRs checked: closed issues mentioning deepspeed,
      merged PRs with "deepspeed" + fix/fixed in title). No clean
      version-bounded bug with a linked fix turned up - mostly usage
      questions (`no_sync` + ZeRO2/3, `main_process_first` + ZeRO3 hang) that
      are architectural limitations or lack a merged fix, not version
      ranges. Zero candidates recorded; moved on per stop condition rather
      than force a low-quality entry.
- [x] **megatron × pytorch** - found `megatron-torch29-dcp-metadata-attr-loss`:
      proved the cause myself directly against source (torch's
      `dataclasses.replace(metadata, ...)` in `filesystem.py`, confirmed
      absent in v2.8.0, present in v2.9.0/2.9.1/2.10.0) rather than trusting
      the issue text alone. PyTorch's own fix PR for this was closed
      *without* merging, so still open with no fixed_in - capped at medium
      confidence despite the strong proof. Rejected - not expressible:
      needs both a Megatron-LM version/usage-detection field (not tracked at
      all - the earlier note about "Megatron's launch flags via T9" turned
      out not to exist in the code; only downstream parallelism knobs like
      sequence_parallel/expert_parallel/context_parallel are tracked) AND a
      way to combine that with framework_version, which is single-purpose
      today.
- [x] **transformers × pytorch** - found
      `transformers-5101-torch-fp8-e8m0-attributeerror`: verified against
      release-tag source (v5.10.1-5.10.3 vs the v5.11.0 fix) that
      `transformers` crashes on import if the torch build lacks
      `torch.float8_e8m0fnu` - notably true of NVIDIA NGC container torch
      builds even when the version string reads "2.7", which the public
      torch==2.7.0 wheel does not. Rejected - not expressible: no
      `transformers` package-version field, and even framework_version
      alone can't distinguish an NGC container torch build from the public
      wheel of the same nominal version.
- [x] **apex × cuda** - found `apex-groupnorm-sm100-cuda126-build-fail`: a
      source-build failure (setup.py requesting sm_100/sm_120 unconditionally,
      breaking `nvcc` on CUDA 12.4-12.7), confirmed by an NVIDIA engineer in
      the issue thread and fixed via a merged PR I diffed directly. Rejected
      - not expressible (no apex usage/version field at all) and lowest
      priority as flagged: it's an install-time build failure, not a
      runtime training bug.

## Tier 3 - exploratory / bigger lift

- [x] **cuDNN × cuda** - checked the live cuDNN 9.24.0 backend release notes
      directly. Found `cudnn-cuda131-layernorm-rmsnorm-perf-regression`
      (LayerNorm/RMSNorm slower on CUDA Toolkit 13.1+ vs 13.0u2, on B200/
      H200/L40S/A100) - unlike every other Hopper/H100 candidate in this
      file, NVIDIA's own known-issue text names **no** untracked
      precondition, so this is fully expressible on cuda_version + gpu_type
      alone with **no new field needed** - genuinely promotable pending a
      severity/wording decision (it's a WARN-grade performance regression,
      not a correctness bug). Capped at medium confidence: "fix planned for
      a future release," no fixed_in yet.
- [ ] **k8s training-operator × torch** / **volcano × torch** - searched
      `kubeflow/trainer` (repo was renamed from `kubeflow/training-operator`)
      and `volcano-sh/volcano` closed issues for torch-version-tied bugs;
      the closest hit (`kubeflow/trainer#2794`, a 2-node rendezvous hang)
      closed with no linked fix PR, so it doesn't meet the fix-linkage bar.
      Zero candidates recorded - still needs an operator/scheduler version
      extraction hook traincheck doesn't have, and still speculative value
      until a cleaner reported incompatibility turns up.

## Per-cell stop condition

Per `README.md`: stop a cell when a full page of search results yields
zero new verified candidates. Move to the next cell rather than paginating
indefinitely on diminishing returns.

## Field-gaps discovered by mining (candidates for a future extraction task)

Not part of the mining pipeline itself, but surfaced by it - if any of
these get built, re-run the corresponding Tier 2 cell against the
now-expressible trigger:

- ~~`accelerate`/`deepspeed`/`transformers`/`apex` package version~~ **BUILT**
  this session: `dependency_constraints` (a `{package: constraint}` dict
  from `extract_lockfile()`) is now wired into every adapter
  (bare/hpc_shell/skypilot/k8s), not just Ray, and `accelerate` +
  `megatron-core` were added to its tracked-package set (`deepspeed`/
  `transformers`/`apex` were already tracked but unreachable since no
  adapter called it). `parse_pinned_version()` + `dependency_constraint()`
  (both in `traincheck.utils`, both exposed to `Rule` conditions via
  `core._SAFE_BUILTINS`) turn a raw constraint string into a comparable
  version tuple. This immediately unblocked 4 previously-rejected findings,
  now promoted: `ACCELERATE-FSDP2-001`, `DEEPSPEED-ZERO2-001`,
  `APEX-GROUPNORM-001`, `MEGATRON-DCP-001` (see
  `tests/test_version_incompat_rules.py`). Only works when a
  requirements.txt/environment.yml/uv.lock/poetry.lock/Pipfile.lock is
  actually near the job's base_dir and the constraint is a single exact
  version (a loose range like ">=1.13,<1.14" can't be collapsed to one
  version and correctly returns `None` rather than guessing).
- DeepSpeed config's `bf16.enabled` flag - still not tracked (blocks
  `deepspeed-0192-zero3-peft-lora-dtype` even with the version field now
  built, since that bug also needs this).
- PEFT/LoRA usage detection - still not tracked (same blocker as above).
- Megatron-LM version specifically - `megatron-core`'s mere *presence* in
  `dependency_constraints` unblocked `MEGATRON-DCP-001` (that bug doesn't
  depend on Megatron's own version), but a rule needing Megatron's actual
  pinned version would still work today via the same field - no further
  gap here.
- `cudnn_version`
- NCCL profiler-plugin usage (no comm_env var or JobSpec field for this -
  it's a C API a training script opts into, invisible to static config
  reading either way; lowest priority of this list, may not be buildable
  at all without executing code)
