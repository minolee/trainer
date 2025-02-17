from requests.exceptions import SSLError
from src.base import BaseConfig
from transformers import AutoTokenizer
__all__ = ["TokenizerConfig"]

class TokenizerConfig(BaseConfig):
    path: str
    
    def __call__(self):
        try:
            return AutoTokenizer.from_pretrained(self.path)
        except SSLError:
            return AutoTokenizer.from_pretrained(self.path, local_files_only=True)