from __future__ import annotations
from pydantic import BaseModel, ConfigDict
__all__ = ["BaseConfig", "CallConfig"]


class BaseConfig(BaseModel):
    """
    base config for all pipeline.
    supports yaml or json load and dump.
    """
    model_config = ConfigDict(extra="allow")
    
    def dump(self, path):
        from src.utils import write_magic
        write_magic(path, self.model_dump())
    
    @classmethod
    def load(cls, path):
        from src.utils import read_magic
        return cls(**read_magic(path)) # type: ignore

class CallConfig(BaseConfig):
    """함수 이름을 불러오고 kwarg를 전달하는데 사용하는 config"""
    name: str 
    """load할 때 사용할 이름. 함수나 class 이름 사용"""

    # any additional config follows