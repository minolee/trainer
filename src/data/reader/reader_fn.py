# add any raw data processing logic here

from __future__ import annotations
from typing import Callable, Iterable
from .prompt import get_prompt
from src.base import create_register_deco, create_get_fn, BaseMessage
from src.utils import read_magic, rank_iter

__all__ = ["get_reader_fn", "list_reader_fn"]

ReaderFn = Callable[[str], Iterable[list[BaseMessage]]]

_reader_fn: dict[str, ReaderFn] = {}
reader_fn = create_register_deco(_reader_fn)

get_reader_fn = create_get_fn(_reader_fn)
def list_reader_fn():
    return _reader_fn.keys()

@reader_fn
@rank_iter
def read_simple(source: str) -> Iterable[list[BaseMessage]]:
    """jsonl file with dialogHistory key"""
    for item in read_magic(source):
        if "dialogHistory" not in item: continue
        yield [BaseMessage(**x) for x in item["dialogHistory"]]
