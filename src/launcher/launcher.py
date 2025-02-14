import accelerate
from src.base import BaseConfig
from typing import Literal
import os
import subprocess
import shutil
import sys
import random
import string
from ..trainer import TrainConfig
from ..utils import parse_nodelist, is_localhost, read_magic, write_magic, concat, load_module
__all__ = ["LauncherConfig"]

class LauncherConfig(BaseConfig):
    """
    accelerate, deepspeed, slurm 등을 사용하여 실행할 때 사용하는 config

    command line에서 --XXX_config 를 통해 파일을 전달하면 해당 파일을 기반으로 XXX를 실행함
    supported: python(default), accelerate, slurm

    config file을 읽어서 해석하고 그에 맞는 command를 실행
    python run.py --accelerate_config 
    """

    run_config: str
    """Runnable config file path"""
    local_rank: int | None = None
    """Distributed 환경을 위한 값. 자동 설정되므로 수동으로 설정하지 마세요."""
    accelerate_config: str | None = None
    """accelerate 기반으로 실행하고 싶을 경우, accelerate config 파일의 path를 입력"""
    deepspeed_config: str | None = None
    """deepspeed 기반으로 실행하고 싶을 경우, deepspeed config 파일의 path를 입력 (미구현)"""

    nodes: str | None = None
    """node를 미리 정해줄 경우 각각의 node마다 process를 실행함. 반드시 accelerate 또는 deepspeed와 함께 사용할 것"""

    is_main: bool = True
    """main process 구분용. 수동으로 설정하지 마세요."""

    @property
    def is_deepspeed(self):
        if self.deepspeed_config:
            return True
        if self.accelerate_config and "deepspeed" in read_magic(self.accelerate_config):
            return True
        return False

    def __call__(self):
        is_main = self.is_main
        is_slurm = "SLURM_JOB_ID" in os.environ
        self.is_main = False
        args = {f"--{k}": v for k in ["mode", "run_config", "local_rank", "is_main"] if (v:=getattr(self, k, None))}
        main_args = lambda: concat(*[[k, v] for k, v in args.items()])
        # assert sum([bool(self.accelerate_config), bool(self.deepspeed_config), bool(self.slurm_config)]) <= 1, "Only one of accelerate, deepspeed, slurm can be used"

        # copy config files to output directory
        
        config = TrainConfig.load(self.run_config)
        
        if self.nodes:
            # 각각의 node마다 수행할 accelerate config를 복사하여 저장
            nodes = parse_nodelist(self.nodes)
        else:
            nodes = ["localhost"]
        if is_slurm:
            print("Running script with SLURM")
            nodes = parse_nodelist(os.environ["SLURM_JOB_NODELIST"])
            print("# of nodes:", len(nodes))
        if is_main:
            output_dir = config.save_dir
            os.makedirs(output_dir, exist_ok=True)
            launch_config = self.model_dump()
            launch_config["run_config"] = f"{output_dir}/run_config.yaml"
            shutil.copy(self.run_config, f"{output_dir}/run_config.yaml")
            # copy custom files
            for f in os.listdir(sdir:=os.path.dirname(self.run_config)):
                if f.endswith(".py"):
                    shutil.copy(os.path.join(sdir, f), output_dir)
            
            if self.accelerate_config:
                accelerate_config = read_magic(self.accelerate_config)
                accelerate_config["num_machines"] = len(nodes)
                accelerate_config["num_processes"] = len(nodes) * 8

                shutil.copy(self.accelerate_config, f"{output_dir}/accelerate_config.yaml")
                launch_config["accelerate_config"] = f"{output_dir}/accelerate_config.yaml"
            if self.deepspeed_config:
                shutil.copy(self.deepspeed_config, f"{output_dir}/deepspeed_config.yaml")
                launch_config["deepspeed_config"] = f"{output_dir}/deepspeed_config.yaml"
            write_magic(f"{output_dir}/launch_config.yaml", launch_config)
        
        if len(nodes) > 1 or not is_localhost(nodes[0]) and not is_slurm:
            # ssh on each node
            procs = []
            host = nodes[0]
            pwd = os.getcwd()
            proc_port = random.randint(10000, 20000)
            for i, n in enumerate(nodes):
                # 각각의 node마다 ssh로 command 실행
                if self.accelerate_config:
                    procs.append(subprocess.Popen(
                        ["ssh", n] 
                        + ["cd", pwd, "&&"]
                        + [sys.executable, "-m", "accelerate.commands.launch"]
                        + ["--machine_rank", str(i)]
                        + ["--main_process_ip", host]
                        + ["--main_process_port", str(proc_port)]
                        + ["--config_file", self.accelerate_config]
                        + ["run.py"] + main_args()
                    ))
                elif self.deepspeed_config:
                    raise NotImplementedError
                
            # procs = [subprocess.Popen(["ssh", node] + _args) for node in nodes]
            for proc in procs:
                proc.wait()
        
        elif any([self.accelerate_config, self.deepspeed_config]) and is_main:
            # run subprocess
            run = []
            if self.accelerate_config:
                conf = read_magic(self.accelerate_config)
                assert conf.get("num_machines", 1) == len(nodes), "Number of nodes should be equal to num_machines"
                shutil.copy(self.accelerate_config, "accelerate_config.yaml")
                run = [sys.executable, "-m", "accelerate.commands.launch", "run.py"]
                
            if self.deepspeed_config:
                run = [sys.executable, "-m", "deepspeed", "run.py"]
            
            else:
                proc = subprocess.Popen(run + main_args())
                proc.wait()
        
        else:
            load_module(os.path.split(self.run_config)[0]) # for debug
            config()

        if is_main:
            print("Training finished")
            