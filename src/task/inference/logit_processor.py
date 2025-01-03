from src.base import create_get_fn
from transformers.generation import LogitsProcessor
__all__ = ["get_logit_processor"]
get_logit_processor = create_get_fn(__name__, type_hint=LogitsProcessor)
