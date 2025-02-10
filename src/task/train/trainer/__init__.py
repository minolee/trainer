from src.utils import create_get_fn
import transformers, trl

from .dpo_trainer import _DPOTrainer

__all__ = ["get_trainer"]

get_trainer = create_get_fn(__name__, transformers, trl, type_hint=transformers.Trainer)
