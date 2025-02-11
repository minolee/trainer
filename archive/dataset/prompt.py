"""(제거 예정) tokenizer의 ChatTemplate를 사용할 것

prompt for various tasks"""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from src.base import BaseMessage
from src.utils import create_get_fn

__all__ = ["get_prompt", "PromptTemplate"]


class PromptTemplate(BaseModel):
    """Message를 LLM Prompt로 변환하는 데 사용하는 템플릿"""

    # tokenizer의 apply_chat_template를 사용할 수 없다면 이걸 사용할 것
    # dictionary + prompt template -> prompt
    # template(dialogues) -> prompt 가 되면 됨
    model_config = ConfigDict(extra="allow")

    bos_token: str | None = None
    eos_token: str | None = None
    start_header_token: str | None = None
    end_header_token: str | None = None

    system_name: str = "system"
    user_name: str = "user"
    assistant_name: str = "assistant"
    system_prompt: str
    def sync_tokenizer(self, tokenizer):
        pass
    
    def inference_header(self, speaker: str):
        raise NotImplemented

    def wrap(self, message: BaseMessage) -> str:
        raise NotImplemented

class Llama31(PromptTemplate):
    """Llama31 Prompt"""
    bos_token: str = "<|begin_of_text|>"
    eos_token: str = "<|eot_id|>"
    start_header_token: str = "<|start_header_id|>"
    end_header_token: str = "<|end_header_id|>\n\n"
    system_prompt: str = """Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant"""

    def inference_header(self, speaker: str):
        return f"{self.start_header_token}{speaker}{self.end_header_token}"

    def wrap(self, message: BaseMessage):
        return f"{self.start_header_token}{message.speaker}{self.end_header_token}{message.message}{self.eos_token}"

class Gemma2(PromptTemplate):
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    start_header_token: str = "<start_of_turn>"
    end_header_token: str = "<end_of_turn>"
    assistant_name: str = "model"
    system_prompt: str = ""
    def inference_header(self, speaker: str) -> str:
        return f"{self.start_header_token}{speaker}\n"
    
    def wrap(self, message: BaseMessage):
        return f"{self.start_header_token}{message.speaker}\n{message.message}{self.end_header_token}"

class Qwen2(PromptTemplate):
    start_header_token: str = "<|im_start|>"
    system_name: str = "system\n"
    user_name: str = "user\n"
    assistant_name: str = "assistant\n"
    end_header_token: str = "<|im_end|>"

    def inference_header(self, speaker: str):
        return f"{self.start_header_token}{speaker}\n"

    def wrap(self, message: BaseMessage):
        return f"{self.start_header_token}{message.speaker}\n{message.message}{self.end_header_token}\n"

get_prompt = create_get_fn(__name__, type_hint=PromptTemplate)
