"""Stub training entrypoint for the Slurm + torchrun + DeepSpeed example."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", help="Path to a DeepSpeed config file.")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    print(f"would train here with deepspeed config: {args.deepspeed}")


if __name__ == "__main__":
    main()
