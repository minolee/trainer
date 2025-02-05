
def concat(*l: list):
    """
    여러 리스트를 하나로 합칩니다.
    """
    result = []
    for x in l:
        result.extend(x)
    return result