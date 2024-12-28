
from functools import wraps

import torch.distributed as dist

__all__ = ["create_register_deco", "rank_iter"]

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


def rank_iter(fn):
    """
    distributed 환경에서 rank와 world size를 받아서 일부를 skip함
    반드시 iterable을 return하는 함수에 사용할 것
    테스트 필요
    """
    d = dist.is_initialized()
    w = d and dist.get_world_size()
    r = w and dist.get_rank()
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        for i, item in enumerate(fn(*args, **kwargs)):
            if not d or i % w == r:
                yield item
    return inner_fn