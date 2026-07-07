"""Tests for gang-scheduling coherence: a Volcano Job's minAvailable vs its
total task replicas, a volcano-scheduled PyTorchJob with no PodGroup, and a
Kueue-queued job with no gang scheduler.
"""

from pathlib import Path

import yaml

from traincheck.adapters.k8s import adapt_k8s
from traincheck.core import RuleEngine
from traincheck.ir import Field
from traincheck.resolve import resolve
from traincheck.rules import BUILTIN_RULES
from traincheck.validator import JobSpec, Validator

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "k8s_crd"


def _engine() -> RuleEngine:
    engine = RuleEngine()
    for rule in BUILTIN_RULES:
        engine.register(rule)
    return engine


def _resolved(value):
    return Field(value=value, status="resolved", source="test")


def _absent():
    return Field(value=None, status="absent")


def _spec(min_available=None, task_replicas_total=None, scheduler_name=None, queue_name=None) -> JobSpec:
    spec = JobSpec()
    spec.min_available = _resolved(min_available) if min_available is not None else _absent()
    spec.task_replicas_total = _resolved(task_replicas_total) if task_replicas_total is not None else _absent()
    spec.scheduler_name = _resolved(scheduler_name) if scheduler_name is not None else _absent()
    spec.queue_name = _resolved(queue_name) if queue_name is not None else _absent()
    return spec


def _fired_ids(spec: JobSpec) -> set:
    result = _engine().check(vars(spec))
    return {v.rule.id for v in result.violations}


# --- GANG-001: Volcano minAvailable vs sum(replicas) --------------------


def test_gang_001_fires_when_min_available_below_replica_total():
    spec = _spec(min_available=4, task_replicas_total=8)

    fired = _fired_ids(spec)

    assert "GANG-001" in fired


def test_gang_001_passes_when_min_available_equals_replica_total():
    spec = _spec(min_available=8, task_replicas_total=8)

    fired = _fired_ids(spec)

    assert "GANG-001" not in fired


def test_gang_001_detail_names_the_values_seen():
    spec = _spec(min_available=4, task_replicas_total=8)

    result = _engine().check(vars(spec))
    violation = next(v for v in result.violations if v.rule.id == "GANG-001")

    assert violation.detail == "minAvailable=4, sum(replicas)=8"


# --- GANG-002: volcano scheduler with no gang/PodGroup indication -------


def test_gang_002_fires_for_volcano_scheduler_without_gang_config():
    spec = _spec(scheduler_name="volcano")

    assert "GANG-002" in _fired_ids(spec)


def test_gang_002_passes_when_min_available_is_present():
    spec = _spec(scheduler_name="volcano", min_available=8, task_replicas_total=8)

    assert "GANG-002" not in _fired_ids(spec)


def test_gang_002_does_not_fire_for_a_non_volcano_scheduler():
    spec = _spec(scheduler_name="default-scheduler")

    assert "GANG-002" not in _fired_ids(spec)


# --- GANG-003: Kueue queue-name label with no gang scheduler ------------


def test_gang_003_fires_for_kueue_queue_without_gang_scheduler():
    spec = _spec(queue_name="team-a-queue")

    assert "GANG-003" in _fired_ids(spec)


def test_gang_003_does_not_fire_when_scheduler_is_volcano():
    spec = _spec(queue_name="team-a-queue", scheduler_name="volcano", min_available=8, task_replicas_total=8)

    assert "GANG-003" not in _fired_ids(spec)


def test_gang_003_does_not_fire_without_a_queue_label():
    spec = _spec()

    assert "GANG-003" not in _fired_ids(spec)


# --- adapt_k8s: Volcano batch Job (minAvailable vs sum(task replicas)) --


def _volcano_job_doc(min_available, task_replicas=(4, 4)):
    containers = [{"name": "worker", "image": "nvcr.io/nvidia/pytorch:24.01-py3", "command": ["torchrun"]}]
    return {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
        "metadata": {"name": "llm-train"},
        "spec": {
            "minAvailable": min_available,
            "tasks": [
                {"name": "worker", "replicas": task_replicas[0], "template": {"spec": {"containers": containers}}},
                {"name": "ps", "replicas": task_replicas[1], "template": {"spec": {"containers": containers}}},
            ],
        },
    }


def _write(tmp_path, doc, name="job.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc))
    return str(path), str(tmp_path)


def test_volcano_job_reads_min_available_and_replica_total(tmp_path):
    doc = _volcano_job_doc(min_available=4, task_replicas=(4, 4))

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.min_available.status == "resolved"
    assert spec.min_available.value == 4
    assert spec.task_replicas_total.status == "resolved"
    assert spec.task_replicas_total.value == 8


def test_volcano_job_min_available_below_replica_sum_fires_end_to_end(tmp_path):
    doc = _volcano_job_doc(min_available=4, task_replicas=(4, 4))

    spec = adapt_k8s(*_write(tmp_path, doc))
    result = Validator().validate_spec(spec)

    assert "GANG-001" in {v.rule.id for v in result.violations}


def test_volcano_job_min_available_equal_to_replica_sum_passes_end_to_end(tmp_path):
    doc = _volcano_job_doc(min_available=8, task_replicas=(4, 4))

    spec = adapt_k8s(*_write(tmp_path, doc))
    result = Validator().validate_spec(spec)

    assert "GANG-001" not in {v.rule.id for v in result.violations}


# --- adapt_k8s: PyTorchJob + PodGroup / Kueue label ----------------------


def _pytorchjob_doc(scheduler_name="volcano", labels=None):
    container = {"name": "pytorch", "image": "nvcr.io/nvidia/pytorch:24.01-py3", "command": ["torchrun"]}
    template = {"spec": {"schedulerName": scheduler_name, "containers": [container]}}
    doc = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": "llm-train"},
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {"replicas": 1, "template": template},
                "Worker": {"replicas": 7, "template": template},
            }
        },
    }
    if labels:
        doc["metadata"]["labels"] = labels
    return doc


def test_pytorchjob_without_podgroup_leaves_min_available_absent(tmp_path):
    doc = _pytorchjob_doc()

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.min_available.status == "absent"
    assert spec.task_replicas_total.value == 8


def test_pytorchjob_reads_min_available_from_matching_podgroup(tmp_path):
    doc = _pytorchjob_doc()
    podgroup = {
        "apiVersion": "scheduling.volcano.sh/v1beta1",
        "kind": "PodGroup",
        "metadata": {"name": "llm-train"},
        "spec": {"minAvailable": 8},
    }
    path, base_dir = _write(tmp_path, doc)
    (tmp_path / "podgroup.yaml").write_text(yaml.safe_dump(podgroup))

    spec = adapt_k8s(path, base_dir)

    assert spec.min_available.status == "resolved"
    assert spec.min_available.value == 8


def test_volcano_pytorchjob_without_gang_config_fires_end_to_end():
    """Uses the real example fixture: schedulerName volcano, no PodGroup
    manifest anywhere in the same directory.
    """
    spec = resolve(str(EXAMPLES_DIR / "pytorchjob.yaml"))
    result = Validator().validate_spec(spec)

    assert "GANG-002" in {v.rule.id for v in result.violations}


def test_queue_name_label_is_read_and_flagged_without_a_gang_scheduler(tmp_path):
    doc = _pytorchjob_doc(scheduler_name="default-scheduler", labels={"kueue.x-k8s.io/queue-name": "team-a-queue"})

    spec = adapt_k8s(*_write(tmp_path, doc))
    result = Validator().validate_spec(spec)

    assert spec.queue_name.value == "team-a-queue"
    assert "GANG-003" in {v.rule.id for v in result.violations}
