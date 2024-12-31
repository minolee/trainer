from src.base import BaseConfig
from transformers import AutoModelForCausalLM
import torch
__all__ = ["ModelConfig"]

class ModelConfig(BaseConfig):
    base_config: str # huggingface model card or path to config file
    # check https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/auto#transformers.AutoConfig.from_pretrained
    weight_path: str | None = None # path to model weight, if None, will use pretrained weight. If set to "scratch", will not load weight
    device: str = "cuda" # device to use

    def __call__(self) -> torch.nn.Module:
        """load model from config"""
        if self.weight_path is not None:
            if self.weight_path == "scratch":
                model = AutoModelForCausalLM.from_config(self.base_config) # does not load weights
            model.load_state_dict(torch.load(self.weight_path))
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_config)
        
        model.to(self.device) # type: ignore
        return model