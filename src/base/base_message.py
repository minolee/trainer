from __future__ import annotations
from pydantic import BaseModel, ConfigDict, field_validator
from enum import Enum

__all__ = ["DataElem", "BaseMessage", "PreferenceMessage"]

class DataElem(BaseModel):
    """학습 또는 inference에 사용하는단위 element. metadata와 message list로 구성"""
    data_source: str | None = None
    data_index: int | None = None
    passages: list[str] | None = None
    elem: list[BaseMessage]

    @field_validator('elem', mode="before")
    def preserve_subclasses(cls, v):
        # 만약 리스트의 아이템이 이미 모델 인스턴스라면 그대로 반환
        if isinstance(v, list):
            return v
        raise ValueError("D must be a list")

class BaseMessage(BaseModel):
    """LLM 학습에 사용하는 system - user - assistant의 대화를 나타내는 message class"""
    
    model_config = ConfigDict(extra="allow")

    speaker: str
    message: str

    
    def __str__(self):
        return f"{self.speaker}: {self.message}"
    
    def __repr__(self):
        return str(self)

class PreferenceMessage(BaseMessage):
    rejected_message: str