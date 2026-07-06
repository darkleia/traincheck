"""Ray Train entrypoint for the distributed training example."""

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def train_loop_per_worker(config):
    print(f"would train here with config: {config}")


def main() -> None:
    ray.init()

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
