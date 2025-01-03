from __future__ import annotations
from src.base import BaseConfig, CallConfig, create_get_fn
from src.data import DataModule
from src.data.reader import ReaderConfig
from src.data.dataset import DatasetConfig
from src.data.dataloader import DataLoaderConfig
from src.model import ModelConfig
from src.tokenizer import TokenizerConfig
from src.env import MODEL_SAVE_DIR
from .loss import get_loss_fn
from pydantic import Field

import torch
import transformers
import trl
import os

get_trainer = create_get_fn(transformers, trl, type_hint=transformers.Trainer)
get_optimizer = create_get_fn(torch.optim, type_hint=torch.optim.Optimizer)

def create_trainer(config: TrainConfig):
    name = config.model_name

    train_args = transformers.TrainingArguments(f"{MODEL_SAVE_DIR}/{name}", **config.training_arguments.model_dump()) # deepspeed init here
    if config.optimizer:
        train_args.set_optimizer(**config.optimizer.model_dump())
    if config.scheduler:
        train_args.set_lr_scheduler(**config.scheduler.model_dump())
    
    datamodule = DataModule(
        config.reader,
        config.dataset,
        config.dataloader,
        config.tokenizer
    )
    base_trainer: type[transformers.Trainer] = get_trainer(config.base_trainer)
    datamodule.prepare_data(["train", "dev"])
    datamodule.setup(["train", "dev"]) # type: ignore
    datamodule.info()
    model = config.model()
    # print model summary
    # 모델의 모든 파라미터를 가져옵니다.
    params = list(model.parameters())

    # 전체 파라미터 수를 계산합니다.
    total_params = sum(p.numel() for p in params)

    # 학습 가능한(gradient를 계산하는) 파라미터 수를 계산합니다.
    trainable_params = sum(p.numel() for p in params if p.requires_grad)

    # 결과를 출력합니다.
    print(f"전체 파라미터 수: {total_params}")
    print(f"학습 가능한 파라미터 수: {trainable_params}")
    loss_fn: torch.nn.Module = get_loss_fn(config.loss)() # type: ignore
    
    class _Trainer(base_trainer):
        def __init__(self):
            super().__init__(model, train_args)
        
        def get_train_dataloader(self):
            return datamodule["train"]
        
        def get_eval_dataloader(self):
            return datamodule["dev"]
        
        def get_test_dataloader(self):
            return datamodule["test"]

        def compute_loss(self, model, batch, *_, **__):
            inp_tensor = {k: v.to(model.device) for k, v in batch.items()}
            label = inp_tensor.pop("label")
            logits = self.model(**inp_tensor).logits
            loss = loss_fn(logits.view(-1, logits.shape[-1]), label.view(-1))
            return loss

    return _Trainer()

class TrainConfig(BaseConfig):

    model_name: str # 모델이 저장될 이름, 이 path에 저장됨
    base_trainer: str | CallConfig = "Trainer"
    
    reader: ReaderConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    tokenizer: TokenizerConfig
    model: ModelConfig

    loss: CallConfig
    optimizer: CallConfig | None = None
    scheduler: CallConfig | None = None

    training_arguments: BaseConfig = Field(default_factory=lambda: BaseConfig())
    """used for hf trainer config. check https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/trainer#transformers.TrainingArguments"""


    def __call__(self):
        # trainer를 return하는 대신 그냥 train process 전체를 총괄해 버리는 것이 깔끔할 듯
        save_dir = os.path.join(MODEL_SAVE_DIR, self.model_name)
        os.makedirs(save_dir, exist_ok=True)
        trainer = create_trainer(self)

        trainer.train()
        self.model.path = save_dir
        self.dump(os.path.join(save_dir, "config.yaml"))
        
        trainer.save_model(save_dir)
        