from pydantic import Field
from .collate_fn import get_collate_fn
from src.base import BaseConfig, CallConfig
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from functools import partial
__all__ = ["DataLoaderConfig"]

class DataLoaderConfig(BaseConfig):
    
    max_length: int = 4096 # TODO 여기다 하는게 아님
    batch_size: int = 32
    num_workers: int = 0 # base: single process
    shuffle: bool = True
    sampler: str | CallConfig | None = None # TODO
    collate_fn: str | CallConfig = "base_collate_fn"
    padding_side: str | None = None

    def __call__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=partial(
                get_collate_fn(self.collate_fn),
                pad_id = 0,
                padding_side = tokenizer.padding_side or self.padding_side
            )
        )