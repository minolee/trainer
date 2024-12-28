from src.base import BaseConfig
import pytest

class TestBaseConfig:
    def test_base_config(self):
        base_config = BaseConfig()
        assert base_config is not None
        assert base_config.model_dump() == {}
        assert base_config.__repr__() == "BaseConfig()"

    def test_base_config_with_kwargs(self):
        base_config = BaseConfig(a=1, b=2) # type: ignore
        assert base_config is not None
        assert base_config.model_dump() == {"a": 1, "b": 2}
        assert base_config.__repr__() == "BaseConfig(a=1, b=2)"

    def test_base_config_with_kwargs_and_override(self):
        base_config = BaseConfig(a=1, b=2) # type: ignore
        base_config = BaseConfig(a=3, b=4) # type: ignore
        assert base_config is not None
        assert base_config.model_dump() == {"a": 3, "b": 4}
        assert base_config.__repr__() == "BaseConfig(a=3, b=4)"

    def test_base_config_with_kwargs_and_override_and_new(self):
        base_config = BaseConfig(a=1, b=2) # type: ignore
        base_config = BaseConfig(a=3, b=4, c=5) # type: ignore
        assert base_config is not None
        assert base_config.model_dump() == {"a": 3, "b": 4, "c": 5}
        assert base_config.__repr__() == "BaseConfig(a=3, b=4, c=5)"

    def test_save_load(self):
        try:
            base_config = BaseConfig(a=1, b=2) # type: ignore
            base_config.dump("test.yaml")
            loaded_config = BaseConfig.load("test.yaml")
            assert loaded_config.model_dump() == {"a": 1, "b": 2}
            assert loaded_config.__repr__() == "BaseConfig(a=1, b=2)"
        finally:
            import os
            if os.path.exists("test.yaml"): os.remove("test.yaml")