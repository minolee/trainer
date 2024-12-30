
from functools import wraps, partial
from typing import TypeVar, Callable
from types import ModuleType
from .base_config import CallConfig

__all__ = ["create_register_deco", "create_get_fn"]

T = TypeVar("T", bound=Callable)

def create_register_deco(registry: dict):
    """
    함수나 class 이름을 가지고 모듈에서 불러오고 싶을 때 사용
    global level에서 dict를 하나 정의한 뒤, 이 함수의 인자로 전달
    이후 함수나 class를 정의할 때, 이 함수를 데코레이터로 사용
    """

    def deco(fn):
        global _reader_fn
        registry[fn.__name__.lower()] = fn
        @wraps(fn)
        def inner_fn(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner_fn
    return deco

def create_get_fn(registry: dict[str, T] | ModuleType, *fallback_registry: dict[str, T] | ModuleType):
    """
    특정 registry에서 이름을 가지고 함수나 class를 불러오는 함수를 생성

    :param registry: 모듈 또는 create_register_deco에 사용한 dict
    :type registry: dict[str, T] | ModuleType
    """
    def get_fn(name: str | CallConfig) -> T | partial[T]:
        """
        이름 또는 CallConfig를 사용하여 함수나 class를 불러옴

        :param name: 불러올 이름. CallConfig를 사용 시 name field의 값을 가져옴
        :type name: str | CallConfig
        :raises ValueError: registry가 dict나 module이 아닌 경우
        :return: 불러온 함수나 class. CallConfig를 사용한 경우 kwargs를 전달하여 partial 함수를 반환
        :rtype: T | partial[T]
        """
        fn_target = name if isinstance(name, str) else name.name
        for reg in [registry, *fallback_registry]:
            if isinstance(reg, dict):
                name_getter = lambda name: reg[name.lower()] # type: ignore
            elif isinstance(reg, ModuleType):
                name_getter = lambda name: getattr(reg, name) # 여기서는 lower()를 사용하지 않음
            else:
                raise ValueError("registry should be dict or module")
            try:
                if isinstance(name, str):
                    return name_getter(name)
                elif isinstance(name, CallConfig):
                    kwargs = name.model_dump()
                    kwargs.pop("name")
                    return partial(name_getter(name.name), **kwargs)
            except:
                pass
        raise ValueError(f"Name {fn_target} not found on every registry")
            
    return get_fn