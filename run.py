from src.env import Accelerator # initialize first
from src.launcher import LauncherConfig

import tyro

if __name__ == "__main__":
    args = tyro.cli(LauncherConfig)
    args()