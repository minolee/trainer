from __future__ import annotations

from src.base import BaseConfig, CallConfig, BaseMessage
from .reader_fn import get_reader_fn
from enum import Enum
from typing import Iterable, TypeVar
from pydantic import field_serializer

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
    default_reader_fn: str | CallConfig | None = None 
    """ReaderElem에 reader_fn이 없을 때 사용할 reader_fn"""
    default_dataset: str | CallConfig | None = None
    """ReaderElem에 dataset이 없을 때 사용할 dataset"""
    default_prompt: str | CallConfig | None = None
    """ReaderElem에 prompt가 없을 때 사용할 prompt"""

class ReaderElem(BaseConfig):
    """개별 파일을 읽어오는 방법 정의"""
    name: str | None = None
    """데이터셋 이름, optional"""
    source: str
    """데이터셋 경로"""
    split: SplitConfig
    """데이터셋 분할 방법"""

    reader_fn: str | CallConfig | None = None
    """Raw data를 Message 형태로 변환하는 함수"""
    prompt: str | CallConfig | None = None
    """Message를 input으로 변환할 때 사용할 prompt"""
    dataset: str | CallConfig | None = None
    """실제 학습 또는 추론에 사용할 dataset class"""

    __loaded: list[list[BaseMessage]] = []
    """loaded data"""

    def read(self) -> Iterable[list[BaseMessage]]:
        """Read raw data from source"""
        assert self.reader_fn is not None, f"reader_fn of {self.name or self.source} is not defined"
        # assert source.prompt is not None, f"prompt of {source.name or source.source} is not defined"
        reader_fn = get_reader_fn(self.reader_fn)
        for elem in reader_fn(self.source): # type: ignore
            self.__loaded.append(elem)
            yield elem

class SplitConfig(BaseConfig):
    """데이터셋을 분할하는 방법 정의"""
    type: DataType = DataType.MIXED
    strategy: str | None = "cycle"
    split_ratio: str | list[float | int] | None = None
    
    @field_serializer("type")
    def serialize_type(self, type: DataType) -> str:
        return type.value

    def parse_split_ratio(self) -> list[int]:
        """
        parse split ratio into list of int
        "8:1:1" -> [8, 1, 1]
        "1:0:0" -> [1, 0, 0]
        "0.8:0.1:0.1" -> [8, 1, 1]
        
        """
        match self.type:
            case DataType.TRAIN:
                return [1, 0, 0]
            case DataType.DEV | DataType.VALIDATION:
                return [0, 1, 0]
            case DataType.TEST | DataType.PREDICT:
                return [0, 0, 1]
            case DataType.MIXED:
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
                line = next(it)
                split_data[idx].append(line)
                c += 1
            except StopIteration:
                break
        return split_data