import deepspeed
import os
import shutil
from .file_util import iter_dir

def convert_checkpoint(path, checkpoint=None, remove_checkpoint_dir_after_save: bool = True):
    if not checkpoint:
        checkpoint_cands = [int(os.path.basename(d).split("-")[1]) for d in os.listdir(path) if d.startswith("checkpoint")]
        checkpoint = max(checkpoint_cands)
    checkpoint_path = os.path.join(path, f"checkpoint-{checkpoint}")
    
    deepspeed.utils.zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_path, path
    )
    
    # remove zero checkpoint
    if remove_checkpoint_dir_after_save:
        for d in os.listdir(path):
            if d.startswith("checkpoint"):
                shutil.rmtree(os.path.join(path, d))