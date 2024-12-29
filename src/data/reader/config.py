from __future__ import annotations

from src.base import BaseConfig, CallConfig
from pathlib import Path
from enum import Enum
from pydantic import field_serializer

class DataType(Enum):
    TRAIN = "train"
    DEV = "dev"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"
    MIXED = "mixed"

class ReaderConfig(BaseConfig):
    sources: list[ReaderElem]

    lazy: bool = False # load only when needed
    default_reader_fn: str | CallConfig | None = None
    default_dataset: str | CallConfig | None = None
    default_prompt: str | CallConfig | None = None

class ReaderElem(BaseConfig):
    name: str | None = None # name of data
    source: str # path to the data source
    split: SplitConfig

    reader_fn: str | CallConfig | None = None # how to read raw file
    prompt: str | CallConfig | None = None # how to convert raw data into middle structure
    dataset: str | CallConfig | None = None # which dataset to use

class SplitConfig(BaseConfig):
    type: DataType = DataType.MIXED
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
            case DataType.DEV:
                return [0, 1, 0]
            case DataType.TEST:
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
