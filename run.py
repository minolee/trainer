from src.env import Accelerator # initialize first

import argparse
from src.task import TrainConfig, InferenceConfig, EvaluationConfig
from src.launcher import LauncherConfig
import subprocess
import shutil
import tyro

if __name__ == "__main__":

    args = tyro.cli(LauncherConfig)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", type=str, choices=["train", "inference", "evaluation"])
    # parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--local_rank", type=int, default=-1) # for distributed

    # parser.add_argument("--accelerate_config", type=str, default=None, help="path to accelerate config")
    # parser.add_argument("--deepspeed_config", type=str, default=None, help="path to deepspeed config")
    # parser.add_argument("--slurm_config", type=str, default=None, help="path to slurm config")
    # args = parser.parse_args()
    
    # if args.accelerate_config is not None:

    #     subprocess.Popen(["accelerate", "launch", "--config_file", args.accelerate_config])

    # config_cls = {
    #     "train": TrainConfig,
    #     "inference": InferenceConfig,
    #     "evaluation": EvaluationConfig
    # }[args.mode]
    # config = config_cls.load(args.config)
    # config()
    args()