from __future__ import annotations

from src.base import BaseConfig, CallConfig, DataElem
from src.base.base_message import PreferenceMessage
from src.utils import world_size, rank_zero_only
from .reader import get_reader
from enum import Enum
from typing import Iterable, TypeVar, Literal
from pydantic import ConfigDict, Field
from datasets import (
    load_dataset, 
    DatasetDict, 
    Dataset, 
    IterableDataset, 
    IterableDatasetDict,
    concatenate_datasets,
    DownloadMode
)
# from deprecated import deprecated
T = TypeVar("T")

class DataType(Enum):
    TRAIN = "train"
    DEV = "dev"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"
    MIXED = "mixed"

class SplitStrategy(Enum):
    CYCLE = "cycle"
    SIZE = "size"
    RANDOM = "random"

class ReaderConfig(BaseConfig):
    """데이터를 읽어오는 방법을 정의하는 config class"""
    sources: list[ReaderElem]
    """데이터를 읽어오는 방법을 정의하는 ReaderElem들"""

    # lazy: bool = False # load only when needed
    reader: str | CallConfig | None = None 
    """ReaderElem에 reader_fn이 없을 때 사용할 reader_fn"""

    def __len__(self):
        return sum(len(source) for source in self.sources)

    def __call__(self):
        """setup all sources"""
        for source in self.sources:
            source.reader = source.reader or self.reader
            source()
    
    def __getitem__(self, key: str) -> Dataset | IterableDataset | None:
        result = []
        for source in self.sources:
            if key in source:
                # result.extend([x.to_dict() for x in source[key]])
                result.append(source[key])
            # raise KeyError(f"{key} is not in the data")
        # return Dataset.from_list(result, features=get_features(self.sources[0].feature)())
        if len(result) == 0:
            return None
        return concatenate_datasets(result)
    
    @rank_zero_only
    def info(self):
        print("###################")
        print("#### DATA INFO ####")
        print("###################")
        
        print("Number of sources:", len(self.sources))
        for k in DataType.__members__.keys():
            try:
                if isinstance(d:=self[k.lower()], Dataset):
                    print(f"{k} split: {len(d)}")
            except:
                pass
        
        print(f"Number of data: {len(self)}")


class ReaderElem(BaseConfig):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """개별 파일을 읽어오는 방법 정의. call하는 경우 config에 맞게 데이터를 읽은 뒤 split을 진행함."""
    name: str | None = None
    """데이터셋 이름, optional"""
    source: str
    """데이터셋 경로"""
    split: str | None = None
    """데이터셋 분할 방법. https://huggingface.co/docs/datasets/v3.2.0/en/loading#slice-splits 참고"""
    limit: int | None = None
    """설정시 데이터셋 총량 제한"""

    reader: str | CallConfig | None = None
    """Raw data를 Message 형태로 변환하는 함수"""

    # feature: str | None = None
    use_cache: bool = False
    # __load_buf: list[DataElem] = Field(default_factory=list, exclude=True)
    dataset: DatasetDict | IterableDatasetDict | None = Field(default=None, exclude=True)
    # split_buf: dict[str, list[DataElem]] = Field(default_factory=dict, exclude=True)
    """loaded data"""

    def __call__(
        self
    ):
        """Read and split data"""
        # read -> split
        assert self.reader is not None, f"reader_fn of {self.name or self.source} is not defined"
        reader = get_reader(self.reader)
        
        dataset = load_dataset(
            path="json",
            name=self.name,
            data_files=self.source,
            split=self.split,
            download_mode=DownloadMode.FORCE_REDOWNLOAD if not self.use_cache else DownloadMode.REUSE_CACHE_IF_EXISTS
        )

        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        elif isinstance(dataset, IterableDataset):
            dataset = IterableDatasetDict({"train": dataset})
        if self.limit and self.limit > 0:
            dataset = DatasetDict({k: dataset[k].select(range(self.limit)) for k in dataset.keys()})
        self.dataset = dataset.map(reader).filter(lambda x: x is not None)
    
    def __len__(self):
        assert self.dataset, "Dataset not initialized"
        return sum(len(x) for x in self.dataset.values())

    def __contains__(self, key: str):
        assert self.dataset, "Dataset not initialized"
        return key in self.dataset

    def __getitem__(self, key: str) -> Dataset | IterableDataset:
        assert self.dataset, "Dataset not initialized"
        return self.dataset[key]

    def info(self):
        assert self.dataset, "Dataset not initialized"
        print(f"Source: {self.source}")
        print(f"Number of data: {len(self)}")

        if all(len(x) != len(self) for x in self.dataset.values()):
            print("Splitted length", {k: len(v) for k, v in self.dataset.items()})
        print(f"Reader: {self.reader.name if isinstance(self.reader, CallConfig) else self.reader}")

# @deprecated
class SplitConfig(BaseConfig):
    """데이터셋을 분할하는 방법 정의"""
    type: DataType = DataType.MIXED
    strategy: str | None = "cycle"
    split_ratio: str | list[float | int] | None = None
    

    def parse_split_ratio(self) -> list[int]:
        """
        parse split ratio into list of int
        "8:1:1" -> [8, 1, 1]
        "1:0:0" -> [1, 0, 0]
        "0.8:0.1:0.1" -> [8, 1, 1]
        
        """
        match self.type:
            case DataType.TRAIN.value:
                return [1, 0, 0]
            case DataType.DEV.value | DataType.VALIDATION.value:
                return [0, 1, 0]
            case DataType.TEST.value | DataType.PREDICT.value:
                return [0, 0, 1]
            case DataType.MIXED.value:
                assert self.split_ratio is not None, "split_ratio is not defined"
                split_ratio = self.split_ratio
                if isinstance(split_ratio, str):
                    split_ratio = [float(x) for x in split_ratio.split(":")]
                    split_ratio = split_ratio[:3] + [0] * (3 - len(split_ratio))
                mval = min(x for x in split_ratio if x > 0)
                split_ratio = [int(x / mval) for x in split_ratio]
                return split_ratio
        raise NotImplementedError(f"split type {self.type} is not implemented")

    def __call__(self, data: Iterable[T]) -> list[list[T]]:
        """
        split data into train / dev / test
        """
        split_data: list[list[T]] = [[] for _ in range(3)]
        c = 0
        idx = 0
        it = iter(data)
        split_ratio = self.parse_split_ratio()
        while True:
            try:
                if c >= split_ratio[idx]:
                    c = 0
                    idx += 1
                    idx %= 3
                    continue
                line = next(it)
                split_data[idx].append(line)
                c += 1
            except StopIteration:
                break
        return split_data