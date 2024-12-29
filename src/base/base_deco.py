
from functools import wraps, partial
from typing import TypeVar, Callable
from types import ModuleType
from .base_config import CallConfig

__all__ = ["create_register_deco", "create_get_fn"]

T = TypeVar("T", bound=Callable)

def create_register_deco(registry: dict):
    # 함수나 class 이름을 가지고 모듈에서 불러오고 싶을 때 사용
    # global level에서 dict를 하나 정의한 뒤, 이 함수의 인자로 전달
    # 이후 함수나 class를 정의할 때, 이 함수를 데코레이터로 사용

    def deco(fn):
        global _reader_fn
        registry[fn.__name__.lower()] = fn
        @wraps(fn)
        def inner_fn(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner_fn
    return deco

def create_get_fn(registry: dict[str, T] | ModuleType):
    def get_fn(name: str | CallConfig) -> T | partial[T]:
        
        if isinstance(registry, dict):
            name_getter = lambda name: registry[name.lower()] # type: ignore
        elif isinstance(registry, ModuleType):
            name_getter = lambda name: getattr(registry, name) # 여기서는 lower()를 사용하지 않음
        else:
            raise ValueError("registry should be dict or module")
        
        if isinstance(name, str):
            return name_getter(name)
        elif isinstance(name, CallConfig):
            kwargs = name.model_dump()
            kwargs.pop("name")
            return partial(name_getter(name.name), **kwargs)
    return get_fn