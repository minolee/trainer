"""main script for train"""

import argparse
from src.train.config import TrainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    config()