# Mining compatibility rules without hallucinating

This directory is the workbench for version-incompatibility rules
(`src/traincheck/rules/version_incompat.py`) - the one rule category an LLM
must never author from memory. Config-coherence rules (tp×pp≠world_size,
minAvailable<sum(replicas), IB disabled on an IB cluster, ...) are authored
directly in `src/traincheck/rules/config_coherence.py` instead; they never
pass through here, because there's no source URL for an internal
contradiction to be confirmed against.

See `PLAN.md` for the prioritized, checkbox-tracked list of which
component/source cells to mine next and why.

## The five disciplines

1. **Extract, never generate.** A rule may only record a claim that appears
   on a page actually fetched in-session. No asserting a version
   relationship from memory. No citation, no fetched page → no candidate.
2. **Quarantine, then promote.** Mined items land in `candidates.jsonl`,
   never straight into `rules_verified.jsonl` or the `rules/` package.
   Promotion happens only after a separate, adversarial verification pass.
3. **Prove the cause, not just the symptom.** Prefer issues closed with a
   linked/merged fix commit, or a release note naming the fix. Record the
   broken range and the fixed version precisely - never widen "2.18.1" to
   "2.18".
4. **Tier by source authority.** NVIDIA release notes / maintainer-confirmed
   GitHub issues with a fix → high confidence. 2+ independent reports →
   medium. A lone forum/SO post → low, and ships as a needs-verification
   advisory, never an ERROR/WARN rule.
5. **Only promote what the resolver can see.** If the trigger needs a field
   traincheck doesn't extract, it can't become an active rule. If the
   trigger is a host fact (driver/kernel/OFED), it becomes a
   needs-verification rule, not a hard pass/fail.

## Pipeline

```
Prompt A (miner, extract-only)  →  candidates.jsonl  →  Prompt B (verifier)  →  rules_verified.jsonl  →  (manual) rules/version_incompat.py
```

Nothing moves to the right without a human (or a fresh, skeptical agent
pass) re-checking it. `rules_verified.jsonl` is still not code - promoting
a verified line into `version_incompat.py` is a separate, deliberate step
so a bad verification pass can't silently ship.

## Candidate schema (one JSON object per line)

Every field is mandatory unless marked optional. Missing `source_url` or
`symptom` makes a candidate invalid.

```jsonc
{
  "id": "nccl-cuda-2183-fix",
  "rule_type": "version_incompat",              // or env_hazard
  "sides": [
    {"component": "nccl", "version_range": ">=2.18.0,<2.18.3"},
    {"component": "cuda", "version_range": ">=12.0,<12.2"}
  ],
  "symptom": "<verbatim error signature, under 15 words>",
  "fixed_in": {"component": "nccl", "version": "2.18.3"},  // optional
  "trigger_field": "software.nccl + software.cuda",         // JobSpec fields, or "host-dependent"
  "host_dependent": false,
  "source_url": "https://...",
  "source_type": "github_issue_with_fix",       // nvidia_release_notes | github_issue | github_pr | vendor_matrix | forum | stackoverflow
  "source_authority": "authoritative",           // authoritative | corroborated | anecdotal
  "corroborating_urls": [],                      // optional, raises confidence
  "confidence": "high",                          // high | medium | low
  "expressible": true,                           // can traincheck evaluate the trigger?
  "status": "candidate",                         // candidate | verified | rejected | needs_remining
  "notes": ""
}
```

- `version_range`: exactly as precise as the source states.
- `symptom`: verbatim, under 15 words.
- `confidence` is derived, not free choice: authoritative-or-fix-linked →
  high; 2+ independent corroborations → medium; single anecdote → low.
- `rejected` vs `needs_remining`: `rejected` means a specific source was
  found and cited that contradicts or fails to support the claim (closed
  `not_planned`, a maintainer's fix guess later walked back, conflicting
  unconfirmed theories, etc.) - it still requires `source_url`. A claim
  that was *never sourced to begin with* (like the quarantined
  NCCL-RING-001) and for which an active search still turns up nothing
  either way stays `needs_remining` even after an unsuccessful mining
  attempt - an empty search result is not itself a citable source, so it
  can't satisfy `rejected`'s source requirement. Record the attempt in
  `notes` either way so the next pass doesn't repeat the same queries.
- `needs_remining` (an extension beyond candidate/verified/rejected): used
  for a claim carried over with no source at all - see below.

## Where to mine

**Authoritative version matrices first**: NVIDIA CUDA Toolkit release
notes, NCCL release notes/changelog, cuDNN support matrix + driver release
notes, NGC container release notes per tag, PyTorch release/compat matrix.

**GitHub issues** (needs the fix-linkage filter): pytorch/pytorch,
NVIDIA/nccl, deepspeedai/DeepSpeed, huggingface/accelerate,
huggingface/transformers, NVIDIA/Megatron-LM, NVIDIA/apex,
Lightning-AI/pytorch-lightning, kubeflow/training-operator, volcano-sh/volcano.
Prefer Closed + linked/merged PR - that gives broken version, fixed version,
and confirmed cause in one place.

**Corroboration tier** (never sole source for high confidence): NVIDIA
developer forums, Stack Overflow, framework discussion boards.

Component pairs worth exhausting: nccl×cuda, nccl×driver, cuda×driver,
cuda×pytorch, pytorch×nccl, deepspeed×pytorch, deepspeed×cuda,
accelerate×deepspeed, accelerate×fsdp, megatron×pytorch, apex×cuda,
transformers×pytorch, cudnn×cuda, k8s-operator×torch.

## Prompt A — the miner (extract-only)

```
You are mining PROVEN compatibility issues for a training-config linter. You may
only record claims that literally appear on a page you fetch in this session. Do
NOT use prior knowledge to assert any version relationship. If you cannot fetch a
supporting page and copy its error text, output nothing for that item.

Task: search <SOURCE> for compatibility/incompatibility issues involving
<COMPONENT>. Use these queries: <QUERIES>. For each promising result, FETCH the
page, then extract a candidate ONLY IF the page states a concrete failure tied to
specific versions.

For each valid candidate, emit one JSON line matching mining/README.md's schema.
Rules: never widen a patch version to a minor family; symptom is a verbatim error
string under 15 words; prefer Closed issues with a linked fix PR/commit and record
fixed_in when named; source_authority is authoritative for NVIDIA release notes or
a maintainer-confirmed fix, corroborated for 2+ independent pages, anecdotal for a
single forum/SO post; set host_dependent true + trigger_field "host-dependent" for
host facts; discard anything lacking source_url or symptom; never invent fixed_in;
one report = one range, never generalize to a family.

Append all candidates to mining/candidates.jsonl. Do not write to rules/. Print a
count and list any results skipped and why.
```

## Prompt B — the verifier/promoter (adversarial re-grounding)

```
You are verifying mined compatibility candidates in mining/candidates.jsonl. Default
stance: skeptical, assume wrong until re-grounded.

For each candidate:
1. Re-fetch source_url. Confirm the symptom text and BOTH version sides actually
   appear. If not, status "rejected", note why, continue.
2. Try to find the fix: search the repo/release notes for the version that
   resolved it. If found, tighten version_range and set fixed_in. If still open
   with no fix, keep it but cap confidence at medium.
3. Try to falsify: search for reports the same versions work fine, or the cause
   was something else. If contradicted, status "rejected".
4. Check expressibility: can traincheck's resolver populate trigger_field from a
   real config? If not, expressible false and status "rejected" (or convert to a
   needs-verification rule if host_dependent).
5. Recompute confidence from authority + corroboration + fix-linkage. Single
   anecdotal source with no corroboration caps at low.
6. Set status "verified" only if steps 1-4 pass. De-duplicate against already
   verified rules (same components + overlapping range = merge, widening
   corroboration, not a new rule).

Write survivors to mining/rules_verified.jsonl. Print verified/rejected/
needs-verification counts and, for each rejected one, the reason. Change nothing
in src/traincheck/rules/ - promotion to code is a separate manual step.
```

## Promotion gate → `rules/version_incompat.py`

A verified candidate becomes a rule only when:
- `status` is `"verified"` and `expressible` is `true`, and
- `confidence` is `high` or `medium` (low/anecdotal ships as a
  needs-verification advisory, never an ERROR/WARN rule), and
- its trigger maps onto `JobSpec` fields the resolver actually populates.

Carry `source_url` and `symptom` into the rule's `message` (or `fix_suggestion`)
so every fired rule can point the user at the original report - and so any
rule a user disputes can be checked against its source in one click.

## Quarantined items

- **`nccl-ring-a100-2.21`** (`status: needs_remining`): the old flat
  `rules.py` shipped `NCCL-RING-001` ("NCCL Ring on A100 clusters >32 nodes
  deadlocks below NCCL 2.21") with no `source_url` at all - it predates this
  pipeline and never went through it. Pulled from the active rule set;
  carried here so it gets re-derived (or dropped) like any other candidate,
  instead of being grandfathered in on no evidence.
