from pydantic import BaseModel
from .base_message import BaseMessage


from typing import Generic, TypeVar

T = TypeVar("T", bound=BaseMessage)

__all__ = ["Instance"]

class Instance(BaseModel, Generic[T]):
    """학습 또는 inference에 사용하는단위 element. metadata와 message list로 구성"""
    data_source: str | None = None
    data_index: int | None = None
    passages: list[str] | None = None
    elem: list[T]
