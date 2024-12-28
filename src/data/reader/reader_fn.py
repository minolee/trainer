# add any raw data processing logic here

from __future__ import annotations
from typing import Callable, Iterable
from .prompt import get_prompt
from src.base import create_register_deco, rank_iter, BaseMessage
from src.utils import read_magic

__all__ = ["get_reader_fn", "list_reader_fn"]

ReaderFn = Callable[[str], Iterable[list[BaseMessage]]]

_reader_fn: dict[str, ReaderFn] = {}
reader_fn = create_register_deco(_reader_fn)

def get_reader_fn(name: str) -> ReaderFn:
    return _reader_fn[name]

def list_reader_fn():
    return _reader_fn.keys()

@reader_fn
@rank_iter
def read_simple(source: str) -> Iterable[list[BaseMessage]]:
    for item in read_magic(source):
        yield [BaseMessage(**x) for x in item["dialogHistory"]]
