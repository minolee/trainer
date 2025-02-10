from __future__ import annotations
from pydantic import BaseModel, ConfigDict, ValidationError
__all__ = ["BaseConfig", "CallConfig"]


class BaseConfig(BaseModel):
    """
    모든 pipeline에 사용되는 config의 base class
    yaml 또는 json 파일로부터 load와 dump 지원.

    __call__을 구현하면 해당 config를 사용하여 각 파이프라인에 필요한 실제 동작이 실행되는 방식으로 구현할 것

    """
    model_config = ConfigDict(extra="allow", use_enum_values=True)

    def dump(self, path):
        from src.utils import write_magic
        write_magic(path, self.model_dump(exclude_none=True))
    
    @classmethod
    def load(cls, path):
        from src.utils import read_magic
        try:
            return cls(**read_magic(path)) # type: ignore
        except ValidationError as e:
            print(f"Error loading {cls.__name__}")
            raise e

class CallConfig(BaseConfig):
    """함수 이름을 불러오고 kwarg를 전달하는데 사용하는 config"""
    name: str 
    """load할 때 사용할 이름. 함수나 class 이름 사용"""

    # any additional config follows

    def __repr__(self):
        return self.name