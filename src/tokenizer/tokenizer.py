from .config import TokenizerConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
__all__ = ["load_tokenizer"]

def load_tokenizer(config: TokenizerConfig):
    return AutoTokenizer.from_pretrained(config.from_pretrained)