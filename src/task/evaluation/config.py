from src.base import BaseConfig
from src.task.inference import InferenceConfig
__all__ = ["EvaluationConfig"]

class EvaluationConfig(BaseConfig):
    pretrained_model: str # model path or pretrained model card on huggingface
    inference_config: InferenceConfig
    