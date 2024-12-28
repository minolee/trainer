from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from .base_deco import create_register_deco
__all__ = ["BaseConfig", "DictConfig"]


class BaseConfig(BaseModel):
    """
    base config for all pipeline.
    supports yaml load and dump using OmegaConf.
    """
    model_config = ConfigDict(extra="allow")

    def dump(self, path):
        from src.utils import write_magic
        write_magic(path, self.model_dump())
    
    @classmethod
    def load(cls, path):
        from src.utils import read_magic
        return cls(**read_magic(path)) # type: ignore

class DictConfig(BaseConfig):
    """base dictionary config"""
    name: str
    # any additional config follows