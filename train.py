"""main script for train"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse

from src.env import MODEL_SAVE_DIR
from src.train.config import TrainConfig


def main(
    config: TrainConfig
):
    save_dir = os.path.join(MODEL_SAVE_DIR, config.model_name)
    os.makedirs(save_dir, exist_ok=True)
    # model = config.model_load_config()

    trainer = config()
    
    trainer.train()
    config.model_load_config.weight_path = str(os.path.join(save_dir, "model.pt"))
    config.dump(os.path.join(save_dir, "config.yaml"))
    
    trainer.save_model(save_dir)
    # torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    
    # TODO deepspeed run check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    main(config)