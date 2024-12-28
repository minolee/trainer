"""prompt for various tasks"""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from src.base import BaseMessage, create_register_deco
# how?
# 지금 굉장히 더러워 보이는데 나중에 깔끔하게 만들어 보자..

__all__ = ["get_prompt", "list_prompt", "PromptTemplate"]

_prompts: dict[str, type[PromptTemplate]] = {}
prompt = create_register_deco(_prompts)

def get_prompt(name: str) -> type[PromptTemplate]:
    return _prompts[name.lower()]

def list_prompt():
    return _prompts.keys()

class PromptTemplate(BaseModel):
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
    
    def inference_header(self, speaker=None):
        raise NotImplemented

    def wrap(self, message: BaseMessage) -> str:
        raise NotImplemented

@prompt
class Llama31(PromptTemplate):
    bot_token: str = "<|begin_of_text|>"
    eot_token: str = "<|eot_id|>"
    start_header_token: str = "<|start_header_id|>"
    end_header_token: str = "<|end_header_id|>\n\n"
    system_prompt: str = """Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant"""

    def inference_header(self, speaker=None):
        return f"{self.start_header_token}{speaker or self.assistant_name}{self.end_header_token}"

    def wrap(self, message: BaseMessage):
        return f"{self.start_header_token}{message.speaker}{self.end_header_token}{message.message}{self.eot_token}"
    