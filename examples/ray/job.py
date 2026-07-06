"""Ray Train entrypoint for the distributed training example."""

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

RUNTIME_ENV = {
    "pip": ["torch==2.3.0", "transformers==4.38.0"],
    "env_vars": {"NCCL_ALGO": "Ring", "NCCL_IB_DISABLE": "0"},
}


@ray.remote(num_gpus=1)
def warmup_gpu():
    print("warming up a GPU worker")


def train_loop_per_worker(config):
    print(f"would train here with config: {config}")


def main() -> None:
    ray.init(runtime_env=RUNTIME_ENV)

    trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=ScalingConfig(
            num_workers=64,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 12},
        ),
        run_config=RunConfig(name="llm-train"),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
