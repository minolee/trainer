from src.base import BaseConfig
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
__all__ = ["ModelConfig"]

class ModelConfig(BaseConfig):
    path: str
    """huggingface model card or path to config file"""

    model_type: str | None = None
    """모델의 타입, 실제 코드에서는 사용하지 않고 기록용으로만 사용함. model init 과정에서 덮어씌워짐"""
    # check https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/auto#transformers.AutoConfig.from_pretrained
    load_weight: bool = True
    """if false, will load model without weights"""
    device: str = "cuda" # device to use

    load_kwargs: BaseConfig = BaseConfig()

    def __call__(self) -> PreTrainedModel:
        """load model from config"""
        if not self.load_weight:
            model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(self.path)
            ) # does not load weights
        else:
            model = AutoModelForCausalLM.from_pretrained(self.path, **self.load_kwargs.model_dump())
        
        model.to(self.device) # type: ignore
        self.model_type = getattr(model.config, "model_type", "unknown")
        return model