from src.base import BaseConfig, CallConfig
from .dataset import get_dataset
from .prompt import get_prompt
from src.base import DataElem
from transformers import PreTrainedTokenizer

__all__ = ["FormatConfig"]

class FormatConfig(BaseConfig):
    """DataElem을 실제 학습 input string으로 가공할 때 사용할 prompt template와 데이터 형태를 정의하는 config class"""
    
    prompt: str | CallConfig
    """DataElem을 실제 학습 input string으로 가공할 때 사용할 prompt template. 모델별로 정의할 것"""
    
    format: str | CallConfig
    """preference, rank, sft 등의 데이터 형태, Trainer에 따라 다르게 정의할 것"""

    def __call__(self, stage: str, data: list[DataElem], tokenizer: PreTrainedTokenizer):
        prompt_cls = get_prompt(self.prompt)() # type: ignore
        dataset_cls = get_dataset(self.format)
        return dataset_cls(data, stage, prompt_cls, tokenizer) # type: ignore
