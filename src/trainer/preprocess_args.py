# trainer의 argument들의 type을 사전 처리. base type을 받아서 가공함
# 이것도 자동화할 수 있는 방법 없으려나?

from ..utils import create_get_fn
get_custom_fn = create_get_fn()

def as_callable(v):
    return get_custom_fn(v)

def reward_funcs(v):
    if isinstance(v, list):
        return list(map(get_custom_fn, v))
    return get_custom_fn(v)

def optimizer(v):
    import torch
    return create_get_fn(torch.optim, v)

def loss(v):
    import torch
    return create_get_fn(torch.nn, torch.nn.functional, v)

