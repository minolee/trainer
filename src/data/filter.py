from src.base import Instance
from src.utils import create_get_fn

__all__ = ["get_filter"]

def filter_fn(elem: Instance) -> bool:
    ...

get_filter = create_get_fn(__name__, type_hint=filter_fn)