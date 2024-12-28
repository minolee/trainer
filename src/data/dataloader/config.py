from src.base import BaseConfig

__all__ = ["DataLoaderConfig"]

class DataLoaderConfig(BaseConfig):
    
    max_length: int = 4096
    batch_size: int = 32
    num_workers: int = -1
    shuffle: bool = True
    sampler: str | None = None
    sampler_kwargs: dict | None = None
    collate_fn: str = "base_collate_fn"
    padding_side: str | None = None