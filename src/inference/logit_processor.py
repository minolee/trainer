from src.base import create_register_deco, create_get_fn
from transformers.generation import LogitsProcessor
__all__ = ["get_logit_processor", "list_logit_processor"]
_logit_processors: dict[str, type[LogitsProcessor]] = {}
logit_processor = create_register_deco(_logit_processors)

get_logit_processor = create_get_fn(_logit_processors)

def list_logit_processor():
    return _logit_processors.keys()
