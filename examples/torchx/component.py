"""Custom torchx component definition for the distributed training example."""

import torchx.specs as specs


def trainer(image: str = "nvcr.io/nvidia/pytorch:24.01-py3", nnodes: int = 8) -> specs.AppDef:
    return specs.AppDef(
        name="llm-train",
        roles=[
            specs.Role(
                name="worker",
                image=image,
                entrypoint="torchrun",
                args=["--nnodes", str(nnodes), "--nproc-per-node", "8", "train.py"],
                num_replicas=nnodes,
            )
        ],
    )
