# TODO support for custom functions

import importlib.util
import os
import sys
from types import ModuleType
__all__ = ["custom_modules", "load_module"]

custom_modules: dict[str, ModuleType] = {}


def import_module_from_path(module_name, file_path):
    # 파일 경로에서 모듈의 spec(스펙)을 생성합니다.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None
    # spec을 바탕으로 모듈 객체를 생성합니다.
    module = importlib.util.module_from_spec(spec)
    
    # 필요하다면, sys.modules에 등록하여 다른 곳에서 import할 수 있도록 합니다.
    custom_modules[module_name] = module
    
    # 모듈의 코드를 실행하여 초기화합니다.
    spec.loader.exec_module(module)
    
    # return module

def load_module(d: str):
    """
    directory에서 모듈을 로드

    :param d: _description_
    :type d: _type_
    """
    for fname in os.listdir(d):
        if fname.endswith(".py"):
            # abs_path = os.path.abspath(os.path.join(d, fname))
            import_module_from_path(fname[:-3], os.path.join(d, fname))
