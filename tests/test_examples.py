"""Sanity checks for the example fixtures under examples/ - one entrypoint
(plus minimal support files) per launcher stack.
"""

import configparser
import json
from pathlib import Path

import yaml

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "examples"
SLURM_DIR = EXAMPLES_ROOT / "slurm"
K8S_DIR = EXAMPLES_ROOT / "k8s_crd"
SKYPILOT_DIR = EXAMPLES_ROOT / "skypilot"
RAY_DIR = EXAMPLES_ROOT / "ray"
BARE_DIR = EXAMPLES_ROOT / "bare"
TORCHX_DIR = EXAMPLES_ROOT / "torchx"
SUBMITIT_DIR = EXAMPLES_ROOT / "submitit"
PBS_DIR = EXAMPLES_ROOT / "pbs"
LSF_DIR = EXAMPLES_ROOT / "lsf"
SGE_DIR = EXAMPLES_ROOT / "sge"
NATIVE_DIR = EXAMPLES_ROOT / "native"


def _read_non_empty(path: Path) -> str:
    assert path.exists(), f"missing fixture: {path}"
    text = path.read_text()
    assert text.strip(), f"empty fixture: {path}"
    return text


# --- Slurm ---


def test_sbatch_script_exists_and_parses_as_text():
    text = _read_non_empty(SLURM_DIR / "train.sbatch")
    assert "#SBATCH" in text


def test_train_py_exists_and_is_non_empty():
    _read_non_empty(SLURM_DIR / "train.py")


def test_requirements_txt_exists_and_is_non_empty():
    _read_non_empty(SLURM_DIR / "requirements.txt")


def test_ds_config_exists_and_parses_as_json():
    config = json.loads(_read_non_empty(SLURM_DIR / "ds_config.json"))
    assert isinstance(config, dict)
    assert config


# --- Kubernetes / Kubeflow Training Operator ---


def test_k8s_pytorchjob_exists_and_parses_as_yaml():
    doc = yaml.safe_load(_read_non_empty(K8S_DIR / "pytorchjob.yaml"))
    assert doc["kind"] == "PyTorchJob"
    assert "Master" in doc["spec"]["pytorchReplicaSpecs"]
    assert "Worker" in doc["spec"]["pytorchReplicaSpecs"]


def test_k8s_configmap_exists_and_parses_as_yaml():
    doc = yaml.safe_load(_read_non_empty(K8S_DIR / "configmap.yaml"))
    assert doc["kind"] == "ConfigMap"
    assert "NCCL_ALGO" in doc["data"]


def test_k8s_dockerfile_exists_and_is_non_empty():
    text = _read_non_empty(K8S_DIR / "Dockerfile")
    assert text.startswith("FROM")


# --- SkyPilot ---


def test_skypilot_task_exists_and_parses_as_yaml():
    doc = yaml.safe_load(_read_non_empty(SKYPILOT_DIR / "task.yaml"))
    assert doc["resources"]["accelerators"]
    assert doc["num_nodes"] == 8


def test_skypilot_requirements_txt_exists_and_is_non_empty():
    _read_non_empty(SKYPILOT_DIR / "requirements.txt")


# --- Ray ---


def test_ray_cluster_yaml_exists_and_parses_as_yaml():
    doc = yaml.safe_load(_read_non_empty(RAY_DIR / "cluster.yaml"))
    assert doc["cluster_name"]
    assert "available_node_types" in doc


def test_ray_job_py_exists_and_is_non_empty():
    text = _read_non_empty(RAY_DIR / "job.py")
    assert "ScalingConfig" in text


def test_ray_requirements_txt_exists_and_is_non_empty():
    _read_non_empty(RAY_DIR / "requirements.txt")


# --- Bare metal (no scheduler) ---


def test_bare_run_sh_exists_and_is_non_empty():
    text = _read_non_empty(BARE_DIR / "run.sh")
    assert "torchrun" in text


def test_bare_hostfile_exists_and_is_non_empty():
    _read_non_empty(BARE_DIR / "hostfile.txt")


# --- torchx ---


def test_torchx_config_exists_and_parses_as_ini():
    text = _read_non_empty(TORCHX_DIR / ".torchxconfig")
    parser = configparser.ConfigParser()
    parser.read_string(text)
    assert parser.sections()


def test_torchx_run_sh_contains_expected_launch_command():
    text = _read_non_empty(TORCHX_DIR / "run.sh")
    assert "torchx run -s slurm dist.ddp -j 8x8" in text


# --- submitit ---


def test_submitit_job_py_exists_and_is_non_empty():
    text = _read_non_empty(SUBMITIT_DIR / "job.py")
    assert "AutoExecutor" in text
    assert "update_parameters" in text


# --- PBS / LSF / SGE ---


def test_pbs_script_exists_and_parses_as_text():
    text = _read_non_empty(PBS_DIR / "train.pbs")
    assert "#PBS" in text


def test_lsf_script_exists_and_parses_as_text():
    text = _read_non_empty(LSF_DIR / "train.lsf")
    assert "#BSUB" in text


def test_sge_script_exists_and_parses_as_text():
    text = _read_non_empty(SGE_DIR / "train.sge")
    assert "#$" in text


# --- traincheck's native schema ---


def test_native_job_yaml_exists_and_parses_as_yaml():
    doc = yaml.safe_load(_read_non_empty(NATIVE_DIR / "job.traincheck.yaml"))
    assert "cluster" in doc
    assert "nccl" in doc
