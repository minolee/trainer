from src.base import BaseConfig

__all__ = ["ModelConfig"]

class ModelConfig(BaseConfig):
    base_config: str # huggingface model card or path to config file
    # check https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/auto#transformers.AutoConfig.from_pretrained
    weight_path: str | None = None # path to model weight, if None, will use pretrained weight. If set to "scratch", will not load weight
    device: str = "cuda" # device to use