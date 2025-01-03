"""prompt for various tasks"""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from src.base import BaseMessage, create_get_fn, Speaker
from typing import Callable
import sys
# how?
# 지금 굉장히 더러워 보이는데 나중에 깔끔하게 만들어 보자..

__all__ = ["get_prompt", "PromptTemplate"]


class PromptTemplate(BaseModel):
    """Message를 LLM Prompt로 변환하는 데 사용하는 템플릿"""
    # dictionary + prompt template -> prompt
    # template(dialogues) -> prompt 가 되면 됨
    model_config = ConfigDict(extra="allow")

    bot_token: str
    eot_token: str
    start_header_token: str
    end_header_token: str

    system_name: str = "system"
    user_name: str = "user"
    assistant_name: str = "assistant"
    system_prompt: str
    def sync_tokenizer(self, tokenizer):
        pass
    
    def inference_header(self, speaker: Speaker):
        raise NotImplemented

    def wrap(self, message: BaseMessage) -> str:
        raise NotImplemented

class Llama31(PromptTemplate):
    """Llama31 Prompt"""
    bot_token: str = "<|begin_of_text|>"
    eot_token: str = "<|eot_id|>"
    start_header_token: str = "<|start_header_id|>"
    end_header_token: str = "<|end_header_id|>\n\n"
    system_prompt: str = """Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant"""

    def inference_header(self, speaker: Speaker):
        return f"{self.start_header_token}{speaker.value}{self.end_header_token}"

    def wrap(self, message: BaseMessage):
        return f"{self.start_header_token}{message.speaker.value}{self.end_header_token}{message.message}{self.eot_token}"

class Gemma(PromptTemplate):
    start_header_token: str = "<start_of_turn>"
    end_header_token: str = "<end_of_turn>"
    assistant_name: str = "model"

    def inference_header(self, speaker: Speaker):
        return f"{self.start_header_token}{speaker.value}\n"
    
    def wrap(self, message: BaseMessage):
        return f"{self.start_header_token}{message.speaker.value}\n{message.message}{self.end_header_token}"

class Qwen2(PromptTemplate):
    start_header_token: str = "<|im_start|>"
    system_name: str = "system\n"
    user_name: str = "user\n"
    assistant_name: str = "assistant\n"
    end_header_token: str = "<|im_end|>"

    def inference_header(self, speaker: Speaker):
        return f"{self.start_header_token}{speaker.value}\n"

    def wrap(self, message: BaseMessage):
        return f"{self.start_header_token}{message.speaker.value}\n{message.message}{self.end_header_token}\n"

get_prompt = create_get_fn(__name__, type_hint=PromptTemplate)
