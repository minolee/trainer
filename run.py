

import argparse
from src.task import TrainConfig, InferenceConfig, EvaluationConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "inference", "evaluation"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config_cls = {
        "train": TrainConfig,
        "inference": InferenceConfig,
        "evaluation": EvaluationConfig
    }[args.mode]
    config = config_cls.load(args.config)
    config()