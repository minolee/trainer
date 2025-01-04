import torch
from src.base import create_get_fn
import sys
__all__ = ["get_collate_fn"]

def collate_fn_type_hint(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int | dict[str, int],
    padding_side: str
) -> dict[str, torch.Tensor]:
    ...

get_collate_fn = create_get_fn(__name__, type_hint=collate_fn_type_hint)


def base_collate_fn(batch: list[dict[str, torch.Tensor]], pad_id, padding_side="left"):
    result = {k: [] for k in batch[0].keys()}
    if isinstance(pad_id, int):
        pad_id = {k: pad_id for k in result.keys()}
    keys = [k for k, v in batch[0].items() if v.dim() > 0]
    max_len = max(x[key].shape[-1] for x in batch for key in keys)
    for item in batch:
        for k, v in item.items():
            pad_tensor = torch.tensor([pad_id[k]] * (max_len - v.shape[-1])).to(v.dtype)
            x = [v]
            x.insert(int(padding_side == "right"), pad_tensor)
            result[k].append(torch.cat(x, dim=0))
    return {k: torch.stack(v) for k, v in result.items()}