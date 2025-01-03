from __future__ import annotations

from src.base import BaseConfig, CallConfig, DataElem
from src.utils import world_size, rank_zero_only
from .reader import get_reader
from enum import Enum
from typing import Iterable, TypeVar
from pydantic import Field

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

    lazy: bool = False # load only when needed
    reader: str | CallConfig | None = None 
    """ReaderElem에 reader_fn이 없을 때 사용할 reader_fn"""

    def __len__(self):
        return sum(len(source) for source in self.sources)

    def __call__(self):
        """setup all sources"""
        for source in self.sources:
            source.reader = source.reader or self.reader
            source()
    
    def __getitem__(self, key: str) -> list[DataElem]:
        result = []
        for source in self.sources:
            if key in source:
                result.extend(source[key])
        return result
    
    @rank_zero_only
    def info(self):
        print("###################")
        print("#### DATA INFO ####")
        print("###################")
        
        print("Number of sources:", len(self.sources))
        for k in DataType.__members__.keys():
            print(f"{k} split: {len(self[k.lower()])}")
        
        print(f"Number of data: {len(self)}")


class ReaderElem(BaseConfig):
    """개별 파일을 읽어오는 방법 정의. call하는 경우 config에 맞게 데이터를 읽은 뒤 split을 진행함."""
    name: str | None = None
    """데이터셋 이름, optional"""
    source: str
    """데이터셋 경로"""
    split: SplitConfig
    """데이터셋 분할 방법"""
    limit: int | None = None
    """설정시 데이터셋 총량 제한"""

    reader: str | CallConfig | None = None
    """Raw data를 Message 형태로 변환하는 함수"""

    # __load_buf: list[DataElem] = Field(default_factory=list, exclude=True)
    split_buf: dict[str, list[DataElem]] = Field(default_factory=dict, exclude=True)
    """loaded data"""

    def __call__(
        self
    ):
        """Read and split data"""
        # read -> split
        assert self.reader is not None, f"reader_fn of {self.name or self.source} is not defined"
        reader = get_reader(self.reader)
        
        load_buf = list(reader(self.source))
        if self.limit and self.limit > 0:
            ws = world_size()
            load_buf = load_buf[:self.limit // ws]
        self.split_buf = {k: v for k, v in zip(["train", "dev", "test"], self.split(load_buf))}
    
    
    def __len__(self):
        return sum(len(x) for x in self.split_buf.values())

    def __contains__(self, key: str):
        return key in self.split_buf

    def __getitem__(self, key: str) -> list[DataElem]:
        return self.split_buf[key]

    def info(self):
        print(f"Source: {self.source}")
        print(f"Number of data: {len(self)}")

        if all(len(x) != len(self) for x in self.split_buf.values()):
            print("Splitted length", {k: len(v) for k, v in self.split_buf.items()})
        print(f"Reader: {self.reader.name if isinstance(self.reader, CallConfig) else self.reader}")

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