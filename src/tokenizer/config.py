from src.base import BaseConfig
from transformers import AutoTokenizer
__all__ = ["TokenizerConfig"]

class TokenizerConfig(BaseConfig):
    from_pretrained: str
    
    def __call__(self):
        return AutoTokenizer.from_pretrained(self.from_pretrained)