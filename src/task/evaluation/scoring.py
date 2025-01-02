from src.base import create_register_deco

__all__ = ["get_evaluation_fn", "list_evaluation_fn"]

_evaluation_fn: dict[str, type] = {}

evaluation_fn = create_register_deco(_evaluation_fn)

def get_evaluation_fn(name: str) -> type:
    return _evaluation_fn[name]

def list_evaluation_fn():
    return _evaluation_fn.keys()