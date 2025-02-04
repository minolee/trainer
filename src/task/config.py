from src.base import BaseConfig

__all__ = ["TaskConfig"]

class TaskConfig(BaseConfig):
    
    def __call__(self):
        raise NotImplementedError

    @property
    def save_dir(self):
        raise NotImplementedError