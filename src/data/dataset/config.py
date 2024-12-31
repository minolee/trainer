from src.base import BaseConfig, CallConfig
from .dataset import get_dataset
from .prompt import get_prompt
from src.base import DataElem
from transformers import PreTrainedTokenizer

__all__ = ["DatasetConfig"]

class DatasetConfig(BaseConfig):
    
    prompt: str | CallConfig
    """DataElem을 실제 학습 input string으로 가공할 때 사용할 prompt template. 모델별로 정의할 것"""
    
    dataset: str | CallConfig
    """Dataset class"""

    def __call__(self, stage: str, data: list[DataElem], tokenizer: PreTrainedTokenizer):
        prompt_cls = get_prompt(self.prompt)()
        dataset_cls = get_dataset(self.dataset)
        return dataset_cls(data, stage, prompt_cls, tokenizer) # type: ignore
