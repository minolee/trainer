from src.base import BaseConfig, DictConfig
from src.data.reader import ReaderConfig
from src.data.dataloader import DataLoaderConfig
from src.model import ModelConfig
from src.tokenizer import TokenizerConfig
from deepspeed import DeepSpeedConfig
from pydantic import Field
class TrainConfig(BaseConfig):
    data_loader_config: ReaderConfig
    data_processor_config: DataLoaderConfig
    tokenizer_config: TokenizerConfig
    model_load_config: ModelConfig # model_config가 안되는거 실화냐

    loss_config: DictConfig
    optimizer_config: DictConfig
    scheduler_config: DictConfig

    # deepspeed_config: DeepSpeedConfig | None = None

    trainer_kwargs: dict = Field(default_factory=dict) # used for trainer config