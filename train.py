"""main script for train"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from lightning import Trainer

from src.data import DataModule
from src.model import BaseModel
from src.train.config import TrainConfig
import torch

def main(
    config: TrainConfig,
    save_dir: str
):
    os.makedirs(save_dir, exist_ok=True)
    config.dump(os.path.join(save_dir, "config.yaml"))
    datamodule = DataModule(
        config.data_loader_config,
        config.data_processor_config,
        config.tokenizer_config
    )

    datamodule.prepare_data()
    datamodule.setup(["train", "dev"]) # type: ignore

    model = BaseModel(
        config.model_load_config,
        train_config = config
    )

    model.setup()

    trainer = Trainer(**config.trainer_kwargs)
    
    trainer.fit(model, datamodule=datamodule)

    
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    
    # TODO deepspeed run check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    main(config, args.save_dir)