"""데이터를 총괄하는 모듈"""
from src.base import CallConfig
from .reader import ReaderConfig
from .dataset import get_format_fn
from src.tokenizer import TokenizerConfig
from src.utils import rank_zero_only
from transformers import PreTrainedTokenizer
from datasets import Dataset, IterableDataset
__all__ = ['DataModule']


class DataModule:
    def __init__(
        self, 
        reader_config: ReaderConfig,
        format_config: str | CallConfig,
        tokenizer_config: TokenizerConfig
    ):
        super().__init__()
        self.reader_config = reader_config
        self.format_config = format_config
        # self.processed: dict[str, Dataset] = {}
        self.tokenizer: PreTrainedTokenizer = tokenizer_config() # type: ignore
    
    def prepare_data(self, stage: str | list[str]):
        print("Preparing data")
        self.reader_config()
        self.reader_config.info()
    
    @rank_zero_only
    def info(self, stage: str | list[str] | None = None):
        if stage is None:
            stage = ["train", "dev", "test"]
        stages = [stage] if isinstance(stage, str) else stage
        # for stage in stages:
        #     if stage in self.prepared:
        #         print(f"Stage: {stage}")
        #         for dataset in self.prepared[stage]:
        #             print(f"Dataset: {dataset.__class__.__name__}")
        #             print(f"Number of data: {len(dataset)}")

    def __getitem__(self, key: str) -> Dataset | IterableDataset | None:
        ds = self.reader_config[key]
        if not ds: return None
        formatter = get_format_fn(self.format_config)
        return ds.map(formatter)
