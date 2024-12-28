from src.base import BaseConfig
from src.data.reader import ReaderConfig
from src.data.dataloader import DataLoaderConfig
from src.model import ModelConfig

class TrainConfig(BaseConfig):
    data_loader_config: ReaderConfig
    data_processor_config: DataLoaderConfig
    model_config: ModelConfig
