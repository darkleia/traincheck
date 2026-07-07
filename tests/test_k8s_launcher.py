"""Tests for the Kubernetes adapter's command-array launcher parsing,
nprocPerNode resolution (spec-level, deprecated elasticPolicy, and the
command itself), world_size computation, and requests-vs-limits handling.
"""

import yaml

from traincheck.adapters.k8s import adapt_k8s


def _container(command, gpu_limit=None, gpu_request=None, env=None):
    resources = {}
    if gpu_limit is not None:
        resources.setdefault("limits", {})["nvidia.com/gpu"] = gpu_limit
    if gpu_request is not None:
        resources.setdefault("requests", {})["nvidia.com/gpu"] = gpu_request

    container = {"name": "pytorch", "image": "nvcr.io/nvidia/pytorch:24.01-py3", "command": command}
    if resources:
        container["resources"] = resources
    if env:
        container["env"] = [{"name": k, "value": v} for k, v in env.items()]
    return container


def _pytorchjob_doc(
    command,
    master_replicas=1,
    worker_replicas=1,
    nproc_per_node=None,
    elastic_nproc_per_node=None,
    gpu_limit=8,
    gpu_request=None,
    env=None,
):
    container = _container(command, gpu_limit=gpu_limit, gpu_request=gpu_request, env=env)
    template = {"spec": {"containers": [container]}}

    spec = {
        "pytorchReplicaSpecs": {
            "Master": {"replicas": master_replicas, "template": template},
            "Worker": {"replicas": worker_replicas, "template": template},
        }
    }
    if nproc_per_node is not None:
        spec["nprocPerNode"] = nproc_per_node
    if elastic_nproc_per_node is not None:
        spec["elasticPolicy"] = {"nProcPerNode": elastic_nproc_per_node}

    return {"apiVersion": "kubeflow.org/v1", "kind": "PyTorchJob", "metadata": {"name": "test"}, "spec": spec}


def _write(tmp_path, doc):
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path), str(tmp_path)


def test_command_array_yields_nnodes_and_nproc(tmp_path):
    command = ["torchrun", "--nnodes=4", "--nproc-per-node=2", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=2, worker_replicas=2, gpu_limit=2)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_kind.status == "resolved"
    assert spec.launcher_kind.value == "torchrun"
    assert spec.launcher_nnodes.status == "resolved"
    assert spec.launcher_nnodes.value == 4
    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 2
    assert spec.launcher_nproc_per_node.source == "k8s:command"


def test_nproc_per_node_from_spec_level_wins_over_command(tmp_path):
    command = ["torchrun", "--nnodes=2", "--nproc-per-node=2", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, nproc_per_node="4", gpu_limit=4)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 4
    assert "spec.nprocPerNode" in spec.launcher_nproc_per_node.source


def test_nproc_per_node_from_deprecated_elastic_policy(tmp_path):
    command = ["torchrun", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, elastic_nproc_per_node=4, gpu_limit=4)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 4
    assert "elasticPolicy" in spec.launcher_nproc_per_node.source


def test_spec_level_nproc_per_node_wins_over_elastic_policy(tmp_path):
    command = ["torchrun", "train.py"]
    doc = _pytorchjob_doc(
        command, master_replicas=1, worker_replicas=1, nproc_per_node="4", elastic_nproc_per_node=8, gpu_limit=4
    )

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_nproc_per_node.value == 4
    assert "spec.nprocPerNode" in spec.launcher_nproc_per_node.source


def test_world_size_is_replicas_times_nproc_per_node(tmp_path):
    command = ["torchrun", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=2, worker_replicas=2, nproc_per_node="4", gpu_limit=4)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.world_size.status == "resolved"
    assert spec.world_size.value == 16  # 4 replicas * 4 nproc


def test_world_size_prefers_nproc_per_node_over_disagreeing_gpu_limit(tmp_path):
    """The bug this fixes: world_size used to be replicas * the GPU
    resource limit even when nprocPerNode said something different.
    """
    command = ["torchrun", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, nproc_per_node="4", gpu_limit=8)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.world_size.value == 8  # 2 replicas * 4 nproc, NOT 2 * 8
    assert "disagrees" in spec.launcher_nproc_per_node.reason
    assert "disagrees" in spec.world_size.reason


def test_gpu_limit_used_as_stand_in_when_nothing_else_resolves_nproc(tmp_path):
    command = ["python", "train.py"]  # no recognized launcher at all
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, gpu_limit=8)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_nproc_per_node.status == "resolved"
    assert spec.launcher_nproc_per_node.value == 8
    assert "stand-in" in spec.launcher_nproc_per_node.reason
    assert spec.world_size.value == 16  # 2 replicas * 8


def test_nproc_per_node_host_dependent_token_is_unknown(tmp_path):
    command = ["torchrun", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, nproc_per_node="auto", gpu_limit=8)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_nproc_per_node.status == "unknown"
    assert spec.launcher_nproc_per_node.reason == "per-node count is host-dependent"
    assert spec.world_size.status == "unknown"


def test_requests_vs_limits_mismatch_is_flagged(tmp_path):
    command = ["torchrun", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, gpu_limit=8, gpu_request=4)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.gpus_per_node.status == "resolved"
    assert spec.gpus_per_node.value == 8
    assert "requests.nvidia.com/gpu" in spec.gpus_per_node.reason
    assert "limits.nvidia.com/gpu" in spec.gpus_per_node.reason


def test_requests_equal_limits_has_no_reason(tmp_path):
    command = ["torchrun", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, gpu_limit=8, gpu_request=8)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.gpus_per_node.value == 8
    assert spec.gpus_per_node.reason == ""


def test_k8s_var_ref_resolves_against_container_env(tmp_path):
    command = ["torchrun", "--node-rank=$(RANK)", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, gpu_limit=1, env={"RANK": "2"})

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_node_rank.status == "resolved"
    assert spec.launcher_node_rank.value == 2


def test_k8s_var_ref_unresolvable_when_injected_later_by_the_operator(tmp_path):
    """MASTER_ADDR is typically injected by the training operator at pod
    creation time, not declared in the container's own env: list - so it
    must come back absent, not the literal "$(MASTER_ADDR)" text.
    """
    command = ["torchrun", "--master-addr=$(MASTER_ADDR)", "train.py"]
    doc = _pytorchjob_doc(command, master_replicas=1, worker_replicas=1, gpu_limit=1)

    spec = adapt_k8s(*_write(tmp_path, doc))

    assert spec.launcher_master_addr.status == "absent"
