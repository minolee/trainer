from __future__ import annotations
import torch
import lightning as pl
from .config import ModelConfig
from transformers import AutoModel, AutoConfig, PreTrainedModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.train import TrainConfig
    from src.inference import InferenceConfig



class BaseModel(pl.LightningModule):
    """Base model for all model pipeline. can be used for sft model without modification"""
    model: torch.nn.Module
    loss_fn: torch.nn.Module
    
    def __init__(self, model_config: ModelConfig, *, train_config: TrainConfig | None = None, inference_config: InferenceConfig | None = None):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.inference_config = inference_config

    def setup(self):
        # load weights
        if self.model_config.weight_path is not None:
            if self.model_config.weight_path == "scratch":
                self.model = AutoModel.from_config(self.model_config) # does not load weights
            
            self.model.load_state_dict(torch.load(self.model_config.weight_path))
        else:
            self.model = AutoModel.from_pretrained(self.model_config.base_config)
        
        self.model.to(self.model_config.device) # type: ignore

        if self.train_config is not None:
            self.loss_fn = getattr(torch.nn, self.train_config.loss_config.name)(**self.train_config.loss_config.model_dump(exclude={"name"}))


    def configure_optimizers(self):
        assert self.train_config is not None
        optimizer_name = self.train_config.optimizer_config.name
        optimizer_kwargs = self.train_config.optimizer_config.model_dump(exclude={"name"})
        optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters(), **optimizer_kwargs)
        return optimizer
    
    def forward(self, *args, **kwargs):
        o_tensor = self.model(*args, **kwargs)
        return o_tensor
    
    def training_step(self, batch, batch_idx):
        inp_tensor = {k: v.to(self.model_config.device) for k, v in batch.items()}
        label = inp_tensor.pop("label")
        o_tensor = self.model(**inp_tensor)
        loss = self.loss_fn(o_tensor, label)
        return loss
    
