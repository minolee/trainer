from functools import wraps

import os
__all__ = ["rank", "world_size", "is_rank_zero", "rank_zero_print", "rank_zero_only", "rank_iter"]

def rank():
    from src.env import Accelerator
    # print("RANK", Accelerator is not None, Accelerator.process_index if Accelerator is not None else None)
    return (not Accelerator) or Accelerator.process_index


def world_size():
    from src.env import Accelerator
    return (not Accelerator) or Accelerator.num_processes


def is_rank_zero():
    from src.env import Accelerator
    # print("RANK", Accelerator is not None, Accelerator.process_index if Accelerator is not None else None) # None이라는데?
    # print(os.environ.get("LOCAL_RANK", 0)) # 얘는 잘 됨
    # print("Accelerator is", Accelerator) # 얘 아까 잘 됐는데?
    # check if this is main process(rank 0), works for all distributed
    # print(os.environ.get("LOCAL_RANK"), rank(), world_size()) # 동작은 잘 하는데, init 시점을 잘 잡아야 함. model init이랑 함께 dist가 동작하는듯?
    return (not Accelerator) or Accelerator.is_local_main_process

def rank_zero_print(*args, **kwargs):
    if is_rank_zero: print(*args, **kwargs)

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