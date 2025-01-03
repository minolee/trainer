from functools import wraps
import os
__all__ = ["rank", "world_size", "is_rank_zero", "rank_zero_only", "rank_iter"]

def rank():
    return max(int(os.environ.get("LOCAL_RANK", 0)), 0)


def world_size():
    return max(int(os.environ.get("WORLD_SIZE", 1)), 1)


def is_rank_zero():
    # check if this is main process(rank 0), works for all distributed
    # print(os.environ.get("LOCAL_RANK"), rank(), world_size()) # 동작은 잘 하는데, init 시점을 잘 잡아야 함. model init이랑 함께 dist가 동작하는듯?
    return os.environ.get("LOCAL_RANK", "0") == "0"



def rank_zero_only(fn):
    """
    distributed 환경에서 rank가 0인 프로세스에서만 실행되도록 하는 데코레이터
    
    TODO return value가 있는 경우에는 어떻게 처리할 것인가?
    context manager로 처리하는 것이 좋을 것 같음
    """
    @wraps(fn)
    def inner(*args, **kwargs):
        if is_rank_zero():
            return fn(*args, **kwargs)
    return inner

def rank_iter(fn):
    """
    distributed 환경에서 rank와 world size를 받아서 일부를 skip함
    반드시 iterable을 return하는 함수에 사용할 것
    테스트 필요
    """
    w = world_size()
    r = rank()
    @wraps(fn)
    def inner_fn(*args, **kwargs):
        for i, item in enumerate(fn(*args, **kwargs)):
            if i % w == r:
                yield item
    return inner_fn