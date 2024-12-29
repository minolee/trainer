import torch.distributed as dist
from functools import wraps

__all__ = ["rank_iter"]

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