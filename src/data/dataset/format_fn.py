"""Basemessage 형태로 저장된 dataset을 각각의 trainer에 맞는 형태로 가공.

https://github.com/huggingface/trl/blob/main/trl/extras/dataset_formatting.py#L78 의 입력 부분으로 가공하는 것임

"""
from src.base import DataElem
from src.utils import autocast, create_get_fn
from .prompt import PromptTemplate
from typing import Any

__all__ = ["get_format_fn"]

def format_fn(data: DataElem, prompt: PromptTemplate) -> dict[str, Any]:
    """format data for model training"""
    ...

@autocast
def format_sft(data: DataElem, prompt: PromptTemplate) -> dict[str, str]:
    result = {
        "messages": []
    }

    return result


@autocast
def format_preference(data: DataElem) -> dict[str, str]:
    """format data for preference learning"""
    result = {}
    # p = []
    # for message in data.elem[:-1]:
    #     p.append(prompt.wrap(message))
    # p.append(prompt.inference_header(data.elem[-1].speaker))

    # result["raw_prompt"] = "".join(p)
    result["prompt"] = [{"role": msg.speaker, "content": msg.message} for msg in data.elem[:-1]]
    result["chosen"] = [{"role": data.elem[-1].speaker, "content": data.elem[-1].message}]
    result["rejected"] = [{"role": data.elem[-1].speaker, "content": data.elem[-1].rejected_message}]

    return result


get_format_fn = create_get_fn(__name__, type_hint=format_fn) # type: ignore