from src.base import BaseConfig, DictConfig

from src.data.reader import ReaderConfig
from src.data.dataloader import DataLoaderConfig
from src.tokenizer import TokenizerConfig
from src.model import ModelConfig

__all__ = ["InferenceConfig"]

class InferenceConfig(BaseConfig):

    pretrained_model: str # model path or pretrained model card on huggingface
    
    data_loader_config: ReaderConfig
    data_processor_config: DataLoaderConfig

    decoder_config: DictConfig

    # deepspeed_config: DeepSpeedConfig | None = None
