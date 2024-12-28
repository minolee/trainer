from src.base import BaseConfig

__all__ = ["TokenizerConfig"]

class TokenizerConfig(BaseConfig):
    from_pretrained: str | None = None
    