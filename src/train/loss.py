from src.base import create_register_deco
import torch

__all__ = ["get_loss_fn", "list_loss_fn"]

_loss_fn: dict[str, type[torch.nn.Module]] = {}

loss_fn = create_register_deco(_loss_fn)

def get_loss_fn(name: str) -> type[torch.nn.Module]:
    if name in _loss_fn:
        return _loss_fn[name]
    try:
        return getattr(torch.nn, name)
    except AttributeError:
        raise ValueError(f"loss function {name} not found")

def list_loss_fn():
    return _loss_fn.keys()

# loss_fn(torch.nn.NLLLoss) # 이렇게 해도 등록 가능
# loss_fn(torch.nn.CrossEntropyLoss)
