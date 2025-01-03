# add any raw data processing logic here

from __future__ import annotations
from typing import Callable, Iterable
from src.base import create_get_fn, BaseMessage, PreferenceMessage, DataElem
from src.utils import read_magic, rank_iter


__all__ = ["get_reader"]

get_reader = create_get_fn(__name__)

@rank_iter
def read_simple(source: str) -> Iterable[DataElem]:
    """jsonl file with dialogHistory key"""
    for i, item in enumerate(read_magic(source)):
        if "dialogHistory" not in item: continue
        yield DataElem(
            data_source=source,
            data_index=i,
            elem=[BaseMessage(**x) for x in item["dialogHistory"]]
        )

@rank_iter
def read_preference(source: str) -> Iterable[DataElem]:
    for i, item in enumerate(read_magic(source)):
        if "dialogHistory" not in item: continue
        if "chosen" not in item: continue
        if "rejected" not in item: continue
        messages = [BaseMessage(**x) for x in item["dialogHistory"][:-1]]
        messages.append(PreferenceMessage(
            speaker=item["dialogHistory"][-1].speaker,
            message=item["chosen"],
            rejected_message=item["rejected"]
        ))
        yield DataElem(
            data_source=source,
            data_index=i,
            elem=messages
        )