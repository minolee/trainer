from __future__ import annotations

from src.base import BaseConfig, CallConfig, DataElem
from src.base.base_message import PreferenceMessage
from src.utils import world_size, rank_zero_only
from .reader import get_reader
from .formatter import get_formatter
from enum import Enum
from typing import Iterable, TypeVar, Literal
from pydantic import ConfigDict, Field
import os
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

class DataConfig(BaseConfig):
    """데이터를 읽어오는 방법을 정의하는 config class"""
    sources: list[DataLoaderElem | str]
    """데이터를 읽어오는 방법을 정의하는 ReaderElem들"""

    # lazy: bool = False # load only when needed
    reader: str | CallConfig | None = None 
    """Default reader function. ReaderElem에 reader를 정의하지 않을 경우 이 reader를 사용함"""

    formatter: str | CallConfig | None = None
    """Default formatter function. ReaderElem에 foramtter를 정의하지 않을 경우 이 formatter를 사용함"""

    def __len__(self):
        return sum(len(source) for source in self.sources)

    def __call__(self):
        """setup all sources"""
        for source in self.sources:
            if isinstance(source, str):
                source = DataLoaderElem(name=source)
            source.reader = source.reader or self.reader
            source.formatter = source.formatter or self.formatter
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

class DataLoaderElem(BaseConfig):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """개별 파일을 읽어오는 방법 정의. call하는 경우 config에 맞게 데이터를 읽은 뒤 split을 진행함."""
    name: str | None = None
    """데이터셋 이름 또는 hf dataset card"""
    source: str | list[str] | None = None
    """데이터셋 경로"""
    split: str | None = None
    """데이터셋 분할 방법. https://huggingface.co/docs/datasets/v3.2.0/en/loading#slice-splits 참고"""
    limit: int | None = None
    """설정시 데이터셋 총량 제한"""

    reader: str | CallConfig | None = None
    """Raw data를 Message 형태로 변환하는 함수"""

    formatter: str | CallConfig | None = None
    """Message를 Trainer에 적합한 형태로 변환하는 함수"""

    use_cache: bool = False
    """Cache 사용 여부"""

    dataset: DatasetDict | IterableDatasetDict | None = Field(default=None, exclude=True)
    """loaded data"""

    def __call__(
        self
    ):
        """Read and split data"""
        # read -> split
        assert self.reader is not None, f"reader_fn of {self.name or self.source} is not defined"
        reader = get_reader(self.reader)
        formatter = get_formatter(self.formatter)
        if self.source is None:
            assert self.name is not None
            self.source = self.name
        if isinstance(self.source, str):
            self.source = [self.source]

        datasets = []
        for source in self.source:
            if os.path.exists(source):
                dataset = load_dataset(
                    path="json",
                    name=self.name,
                    data_files=source,
                    split=self.split,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD if not self.use_cache else DownloadMode.REUSE_CACHE_IF_EXISTS
                )
            else:
                dataset = load_dataset(
                    path=source,
                    split=self.split,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD if not self.use_cache else DownloadMode.REUSE_CACHE_IF_EXISTS
                )
            
            datasets.append(dataset)
        
        def proc(dataset):
            if isinstance(dataset, Dataset):
                dataset = DatasetDict({dataset.split: dataset})
            elif isinstance(dataset, IterableDataset):
                dataset = IterableDatasetDict({dataset.split: dataset})
            if self.limit and self.limit > 0:
                dataset = DatasetDict({k: dataset[k].select(range(self.limit // len(datasets))) for k in dataset.keys()})
            dataset.map(reader).filter(lambda x: x is not None).map(formatter)
            return dataset
        datasets = list(map(proc, datasets))
        
        self.dataset = DatasetDict()
        for dataset in datasets:
            for k, v in dataset.items():
                if k in self.dataset:
                    self.dataset[k] = concatenate_datasets([self.dataset[k], v])
                else:
                    self.dataset[k] = v
    
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
