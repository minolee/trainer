import inspect
from functools import wraps
from pydantic import BaseModel

from functools import wraps, partial
from typing import TypeVar, Callable, Mapping
from types import ModuleType
from ..base.base_config import CallConfig
import sys
__all__ = [
    'create_register_deco', 'create_get_fn', 'drop_unused_args', 'autocast'
]

T = TypeVar("T", bound=Callable)

def create_register_deco(registry: dict):
    """
    Deprecated: 이 decorator를 사용하면 pickling이 안되고 multiprocessing이 안돼서 num_worker같은 option이 안먹힘.
    registry dictionary를 별도로 만드는 대신 create_get_fn(sys.modules[__name__]) 으로 우회 가능

    
    함수나 class 이름을 가지고 모듈에서 불러오고 싶을 때 사용
    global level에서 dict를 하나 정의한 뒤, 이 함수의 인자로 전달
    이후 함수나 class를 정의할 때, 이 함수를 데코레이터로 사용
    
    """

    def deco(fn):
        registry[fn.__name__.lower()] = fn
        @wraps(fn)
        def inner_fn(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner_fn
    return deco


def create_get_fn(*name: str | ModuleType, type_hint: T | None = None) -> Callable[[str | CallConfig], T]:
    """
    특정 registry에서 이름을 가지고 함수나 class를 불러오는 함수를 생성

    >>> get_fn = create_get_fn(module_name)
    >>> inst = get_fn(instance_name)


    함수에 대한 type hint를 지정하려면 typing의 Callable이나 Protocol을 사용하는 대신, dummy 함수를 만든 뒤 해당 함수를 type_hint에 전달해 주세요
    
    :param registry: 모듈 또는 create_register_deco에 사용한 dict
    :param type_hint: 불러올 함수나 class의 type hint, optional
    :type registry: dict[str, T] | ModuleType
    """

    registry = [sys.modules[x] if isinstance(x, str) else x for x in name]
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
        for reg in registry:
            if isinstance(reg, dict):
                name_getter = lambda name: reg[name] # type: ignore
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
    return get_fn # type: ignore


def drop_unused_args(fn, kwargs):
    """
    fn: 호출할 대상 함수
    kwargs: 함수 A에게 전달할 keyword 인자들을 담은 dict

    (ChatGPT generated)
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    # 함수 A가 **kwargs를 허용하는지 확인
    accepts_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in params.values()
    )

    if accepts_var_keyword:
        # A가 추가 keyword 인자를 받을 수 있으므로,
        # 딕셔너리 전체를 전달해도 문제가 없습니다.
        filtered_kwargs = kwargs
    else:
        # A가 **kwargs를 받지 않으므로, A의 인자 목록에 포함된 이름만 전달합니다.
        allowed_names = {
            name for name, param in params.items()
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY
            )
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_names}

    # args에는 함수 A의 *args에 해당하는 인자들을 넣을 수 있습니다.
    return filtered_kwargs

def autocast(fn):
    sig = inspect.signature(fn)
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 전달된 인자들을 함수 시그니처에 바인딩
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # 각 파라미터에 대해 확인
        for name, param in sig.parameters.items():
            annotation = param.annotation
            # annotation이 없는 경우는 넘어감
            if annotation is inspect.Parameter.empty:
                continue
            
            # annotation이 타입이며, pydantic.BaseModel의 서브클래스인지 확인
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                value = bound_args.arguments.get(name)
                # 전달된 값이 dict라면 모델로 변환
                if isinstance(value, Mapping):
                    bound_args.arguments[name] = annotation(**value)
        
        # 변환된 인자를 사용하여 원래 함수를 호출
        return fn(*bound_args.args, **bound_args.kwargs)
    
    return wrapper