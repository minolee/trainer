import inspect

__all__ = ['drop_unused_args']

def drop_unused_args(fn, kwargs):
    """
    fn: 호출할 대상 함수
    kwargs: 함수 A에게 전달할 keyword 인자들을 담은 dict
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    # 함수 A가 **kwargs를 허용하는지 확인
    accepts_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in params.values()
    )

    if accepts_var_keyword:
        # A가 추가 keyword 인자를 받을 수 있으므로,
        # 딕셔너리 전체를 전달해도 문제가 없습니다.
        filtered_kwargs = kwargs
    else:
        # A가 **kwargs를 받지 않으므로, A의 인자 목록에 포함된 이름만 전달합니다.
        allowed_names = {
            name for name, param in params.items()
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY
            )
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_names}

    # args에는 함수 A의 *args에 해당하는 인자들을 넣을 수 있습니다.
    return filtered_kwargs