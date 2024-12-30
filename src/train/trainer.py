from .config import TrainConfig
from .loss import get_loss_fn
import transformers, trl
from transformers import Trainer

from src.data import DataModule
from src.base import create_get_fn
from typing import Callable
import torch

__all__ = ["create_trainer"]

get_trainer = create_get_fn(transformers, trl) # type: ignore


def create_trainer(
    config: TrainConfig
) -> Trainer:
    datamodule = DataModule(
        config.data_loader_config,
        config.data_processor_config,
        config.tokenizer_config
    )
    base_trainer: type[Trainer] = get_trainer(config.base_trainer)
    datamodule.prepare_data()
    datamodule.setup(["train", "dev"]) # type: ignore

    model = config.model_load_config()
    loss_fn: torch.nn.Module = get_loss_fn(config.loss_config)() # type: ignore
    class _Trainer(base_trainer):
        
        def get_train_dataloader(self):
            return datamodule.train_dataloader()
        
        def get_eval_dataloader(self):
            return datamodule.val_dataloader()
        
        def get_test_dataloader(self):
            return datamodule.test_dataloader()

        def training_step(self, model, batch, *_):
            inp_tensor = {k: v.to(model.device) for k, v in batch.items()}
            label = inp_tensor.pop("label")
            logits = self.model(**inp_tensor).logits
            loss = loss_fn(logits.view(-1, logits.shape[-1]), label.view(-1))
            return loss
    
    return _Trainer(model)
        