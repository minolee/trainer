# add any raw data processing logic here

from __future__ import annotations
from typing import Iterable, Any
from src.base import BaseMessage, PreferenceMessage, DataElem
from src.utils import create_get_fn


__all__ = ["get_reader"]

# 250121: removed rank_iter - accelerate에서 알아서 split해서 넣어줌

def reader_type(source: Any) -> dict | None:
    ...

get_reader = create_get_fn(__name__, type_hint=reader_type) # 이게되네

def read_simple(source: dict) -> dict | None:
    """jsonl file with dialogHistory key"""
    if "dialogHistory" not in source: return
    return DataElem(
        elem=[BaseMessage(**x) for x in source["dialogHistory"]]
    ).model_dump()


def read_preference(source: dict) -> dict | None:
    if "dialogHistory" not in source: return
    if "chosen" not in source: return
    if "rejected" not in source: return
    messages = [BaseMessage(**x) for x in source["dialogHistory"][:-1]]
    messages.append(PreferenceMessage(
        speaker=source["dialogHistory"][-1]["speaker"],
        message=source["chosen"]["message"],
        rejected_message=source["rejected"]["message"]
    ))
    return DataElem(
        elem=messages
    ).model_dump()