from .rank import rank, is_rank_zero
import inspect

__all__ = ["pprint", "rank_zero_pprint", "rank_zero_print"]

def rank_zero_print(*args, **kwargs):
    if is_rank_zero: print(*args, **kwargs)

def rank_zero_pprint(*args, **kwargs):
    if is_rank_zero(): pprint(*args, **kwargs)

def pprint(*args, **kwargs):
    frame = inspect.currentframe().f_back.f_back
    filename = frame.f_code.co_filename
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    print(f"[Rank {rank()}] [{filename}:{line_number}@{function_name}]:", *args, **kwargs)