from src.base import BaseConfig
from typing import Literal
import subprocess
import shutil
from src.task import TrainConfig, InferenceConfig, EvaluationConfig
class LauncherConfig(BaseConfig):
    mode: Literal["train", "inference", "eval"]
    config: str
    local_rank: int | None = None
    accelerate_config: str | None = None
    deepspeed_config: str | None = None
    slurm_config: str | None = None


    def __call__(self):
        args = {f"--{k}": str(v) for k, v in self.dict().items()}
        if self.accelerate_config:
            del args["--accelerate_config"]
            shutil.copy(self.accelerate_config, "accelerate_config.yaml")
            _args = ["accelerate", "launch", "--config_file", self.accelerate_config, "run.py"]
            for k, v in args.items():
                _args.append(k)
                _args.append(v)
            subprocess.Popen(_args)
            return
        
        config_cls = {
            "train": TrainConfig,
            "inference": InferenceConfig,
            "evaluation": EvaluationConfig
        }[self.mode]
        config = config_cls.load(self.config)
        config()