# traincheck

![PyPI version](https://img.shields.io/pypi/v/traincheck-lint.svg)

A linter for distributed GPU training configs. Point it at your job config, whatever launcher you use, and it flags the misconfigs that either crash the job at step 0 or quietly tank your throughput, before you burn a multi-node allocation finding out the hard way.

* [GitHub](https://github.com/darkleia/traincheck/) | [PyPI](https://pypi.org/project/traincheck-lint/)
* PyPI package is `traincheck-lint` (the plain `traincheck` name is already taken by an unrelated project) — the CLI command itself is still just `traincheck`.
* Created by [Victoria Besedina](-) | GitHub [@darkleia](https://github.com/darkleia)
* MIT License

## why this exists

Multi-node GPU training has a handful of footguns that nobody warns you about until you've been burned once like NCCL Ring algo deadlocking on old NCCL versions with >32 A100 nodes, InfiniBand accidentally disabled via an env var, GDR level too low on H100s, tensor/pipeline/data-parallel degrees that don't actually multiply out to your GPU count, dataloader workers too low for 8-GPU nodes, checkpoint intervals too sparse for a big run. traincheck knows about these and checks your config for them before you submit.

It's also honest about what it can't know statically: driver version, whether OFED is installed, whether the nvidia-peermem kernel module is loaded. None of that lives in a config file. Instead of guessing, traincheck flags these separately as needs verification and gives you the exact shell command to check yourself, or pass `--probe-host` and it'll just check them on whatever machine you're running from.

## who this is for

Anyone submitting distributed training jobs to a shared GPU cluster who's tired of discovering a misconfig 20 minutes into a job. ML infra folks, anyone maintaining training scripts across a team, anyone who's ever had a multi-node run hang because NCCL picked the wrong network interface.

## supported stacks

- Slurm (sbatch scripts)
- Kubernetes / Kubeflow (PyTorchJob, MPIJob, TFJob, Volcano Job, plain batch Job)
- SkyPilot
- Ray (cluster.yaml + job.py)
- torchx
- submitit
- bare metal (no scheduler, just a launch script)
- traincheck's own native YAML schema, if you'd rather write config directly

DeepSpeed configs and Hydra-composed configs get pulled in automatically wherever your launch command references them, whatever the underlying scheduler.

## usage

```bash
pip install traincheck-lint   # or: uv tool install traincheck-lint

traincheck check path/to/your/job/config
```

Working on traincheck itself instead? `uv tool install --editable .` from a clone gets you a live-updating local install.

That's it — traincheck figures out which stack you're pointing it at on its own. Add `--json` for machine-readable output, or `--probe-host` if you want it to actually check driver/kernel/OFED/peermem on the current machine (only meaningful if that machine is representative of where the job actually runs).

Exit code is 1 if there's a real violation, 0 otherwise — verification items don't fail the run, they're just flagged for you to go check.

## status

Still early. Detection exists for PBS/LSF/SGE but there's no adapter for them yet, Accelerate configs aren't parsed, and a few Slurm GPU-request flag spellings (`--gres=`, `--gpus-per-task`, etc.) aren't handled yet either. Issues and PRs welcome.

## development

```bash
uv run pytest   # tests
just qa         # format, lint, type check, test
```
