from pydantic import BaseModel, field_validator
from enum import Enum

__all__ = ["Speaker", "BaseMessage", "PreferenceMessage"]

class Speaker(Enum):
    """Speaker 정보"""
    SYSTEM = "System"
    USER = "User"
    ASSISTANT = "Assistant"
    MIDM = "Midm"
    @property
    def type(self) -> str:
        match self:
            case Speaker.MIDM:
                return "Assistant"
        return self.value

class BaseMessage(BaseModel):
    """LLM 학습에 사용하는 system - user - assistant의 대화를 나타내는 message class"""
    speaker: Speaker
    message: str

    @field_validator("speaker", mode="before")
    @classmethod
    def check_speaker(cls, v):
        return Speaker[v.upper()]

    def __str__(self):
        return f"{self.speaker}: {self.message}"
    
    def __repr__(self):
        return str(self)

class PreferenceMessage(BaseMessage):
    content_b: str