from __future__ import annotations
from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["BaseMessage", "PreferenceMessage"]



class BaseMessage(BaseModel):
    """LLM 학습에 사용하는 system - user - assistant의 대화를 나타내는 message class"""
    
    model_config = ConfigDict(extra="allow")

    speaker: str
    message: str

    @field_validator("speaker")
    def validate_speaker(cls, v):
        v = v.lower()
        if v == "midm":
            v = "assistant"
        return v


    def __str__(self):
        return f"{self.speaker}: {self.message}"
    
    def __repr__(self):
        return str(self)

class PreferenceMessage(BaseMessage):
    rejected_message: str | None = None


