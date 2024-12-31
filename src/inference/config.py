from __future__ import annotations

from src.base import BaseConfig, CallConfig

from src.data.reader import ReaderConfig
from src.data.dataloader import DataLoaderConfig
from src.tokenizer import TokenizerConfig
from src.model import ModelConfig

__all__ = ["InferenceConfig"]

class InferenceConfig(BaseConfig):

    pretrained_model: str # model path or pretrained model card on huggingface

    data_loader_config: ReaderConfig
    data_processor_config: DataLoaderConfig

    generation_config: GenerationConfig

    output_config: str

    # deepspeed_config: DeepSpeedConfig | None = None

class GenerationConfig(BaseConfig):
    """Huggingface generation config를 따름. 정의되지 않은 값은 모델의 generation config를 가져온다.
    
    https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig 참고"""
    # 굳이 모든 field를 가져와서 정의해 둬야 하나? 그냥 extra allow 되어 있으니까 빈 class로 놔둬도 되는 것 아닐까?
    # max_length: int | None = None
    # max_new_tokens: int | None = None
    # min_length: int | None = None
    # min_new_tokens: int | None = None
    # early_stopping: bool | None = None
    # max_time: float | None = None
    # stop_strings: str | list[str] | None = None

    # do_sample: bool | None = None
    # num_beams: int | None = None
    # num_beam_groups: int | None = None
    # penalty_alpha: float | None = None
    # dola_layers: str | list[int] | None = None

    # use_cache: bool | None = None
    # cache_implementation: str | None = None
    # cache_config: dict | None = None
    # return_legacy_cache: bool | None = None

    # temperature: float | None = None
    # top_k: int | None = None
    # top_p: float | None = None
    # min_p: float | None = None


    def model_dump(self):
        return super().model_dump(exclude_none=True)