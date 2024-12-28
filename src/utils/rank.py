
__all__ = ['is_rank_zero']
def is_rank_zero():
    # check if this is main process(rank 0), works for all distributed
    try:
        import deepspeed
        return deepspeed.utils.rank_zero_only.rank == 0
    except ImportError:
        pass
    try:
        import torch
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        else:
            return True
    except ImportError:
        pass
    return True