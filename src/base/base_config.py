from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from .base_deco import create_register_deco
__all__ = ["BaseConfig", "Config"]

_config = {}
Config = create_register_deco(_config)

def get_config(name: str) -> type[BaseConfig]:
    return _config[name.lower().replace("config", "") + "config"]


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

    