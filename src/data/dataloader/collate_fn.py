import torch
from src.base import create_register_deco, create_get_fn
__all__ = ["get_collate_fn", "list_collate_fn"]
_collate_fn = {}

collate_fn = create_register_deco(_collate_fn)


get_collate_fn = create_get_fn(_collate_fn)

def list_collate_fn():
    return _collate_fn.keys()

@collate_fn
def base_collate_fn(batch: list[dict[str, torch.Tensor]], pad_id, padding_side="left"):
    result = {k: [] for k in batch[0].keys()}
    if isinstance(pad_id, int):
        pad_id = {k: pad_id for k in result.keys()}
    max_len = max(x["input_ids"].shape[-1] for x in batch)
    for item in batch:
        for k, v in item.items():
            pad_tensor = torch.tensor([pad_id] * (max_len - v.shape[-1])).to(v.dtype)
            x = [v]
            x.insert(int(padding_side == "right"), pad_tensor)
            result[k].append(torch.cat(x, dim=0))
    return {k: torch.stack(v) for k, v in result.items()}