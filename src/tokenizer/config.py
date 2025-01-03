from src.base import BaseConfig
from transformers import AutoTokenizer
__all__ = ["TokenizerConfig"]

class TokenizerConfig(BaseConfig):
    path: str
    
    def __call__(self):
        return AutoTokenizer.from_pretrained(self.path)