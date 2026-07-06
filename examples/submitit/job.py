"""Submitit entrypoint - wraps Slurm submission in Python."""

import submitit


def train():
    print("would train here")


def main() -> None:
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        nodes=8,
        tasks_per_node=8,
        gpus_per_node=8,
        cpus_per_task=16,
        mem_gb=640,
        timeout_min=1440,
        slurm_partition="gpu",
        slurm_constraint="h100",
        slurm_additional_parameters={"nccl-algo": "Ring"},
    )
    job = executor.submit(train)
    print(job.job_id)


if __name__ == "__main__":
    main()
