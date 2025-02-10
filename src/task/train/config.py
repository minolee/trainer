from __future__ import annotations

import accelerate
from src.base import BaseConfig, CallConfig
from src.data import DataModule
from src.data.reader import ReaderConfig, PromptTemplate, get_prompt
from src.data.dataset import FormatConfig
from src.data.dataloader import get_collate_fn, DataLoaderConfig
from src.model import ModelConfig
from src.tokenizer import TokenizerConfig
from src.env import MODEL_SAVE_DIR
from src.utils import world_size, is_rank_zero, rank, drop_unused_args, create_get_fn

from .trainer import get_trainer
from ..config import TaskConfig
from pydantic import Field

import torch
import transformers
import trl
import os
import inspect


get_optimizer = create_get_fn(torch.optim, type_hint=torch.optim.Optimizer)

def create_trainer(config: TrainConfig):
    """Config를 사용하여 Trainer를 생성합니다. Deepspeed config가 있을 경우 필요한 설정을 추가합니다."""
    
    kwargs = {} # PPO같은 일부 trainer들은 model, arg 순서로 받지 않아서 kwargs로 넘겨줌

    name = config.model_name
    per_device_train_batch_size = getattr(config.training_arguments, "per_device_train_batch_size", 8)
    # config.dataloader.batch_size = per_device_train_batch_size
    if hasattr(config.training_arguments, "deepspeed"):
        config.training_arguments.deepspeed["train_micro_batch_size_per_gpu"] = per_device_train_batch_size # type: ignore
        # config.training_arguments.load_best_model_at_end = True
    # set trainer and training argument class
    
    base_trainer: type[transformers.Trainer] = get_trainer(config.base_trainer)
    argument_cls = inspect.signature(base_trainer.__init__).parameters["args"].annotation
    if not inspect.isclass(argument_cls): # maybe union or optional type
        argument_cls = argument_cls.__args__[0]
    train_args: transformers.TrainingArguments = argument_cls(
        f"{MODEL_SAVE_DIR}/{name}", 
        **drop_unused_args(argument_cls.__init__, config.training_arguments.model_dump())
    ) # deepspeed init here
    
    
    if config.optimizer:
        train_args.set_optimizer(**config.optimizer.model_dump())
    if config.scheduler:
        train_args.set_lr_scheduler(**config.scheduler.model_dump())
    # if config.loss:
    #     kwargs["compute_loss_func"] = get_loss_fn(config.loss)()
    # load model
    model = config.model()
    tokenizer = config.tokenizer()
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs["model"] = model
    kwargs["processing_class"] = tokenizer
    kwargs["args"] = train_args
    
    if config.ref_model:
        ref_model = config.ref_model()
        ref_model.eval()
        kwargs["ref_model"] = ref_model

    if config.reward_model:
        kwargs["reward_model"] = config.reward_model()
    if config.dataloader.collate_fn:
        kwargs["data_collator"] = get_collate_fn(config.dataloader.collate_fn)
    
    # load data
    datamodule = DataModule(
        config.reader,
        config.format,
        config.tokenizer
    )

    datamodule.prepare_data(["train", "dev"])
    datamodule.info()
    kwargs["train_dataset"] = datamodule["train"]
    kwargs["eval_dataset"] = datamodule["dev"]
    
    
    # print model summary
    # 모델의 모든 파라미터를 가져옵니다.
    # if world_size() == 1:
    #     params = list(model.parameters())

    #     # 전체 파라미터 수를 계산합니다.
    #     total_params = sum(p.numel() for p in params)

    #     # 학습 가능한(gradient를 계산하는) 파라미터 수를 계산합니다.
    #     trainable_params = sum(p.numel() for p in params if p.requires_grad)

    #     # 결과를 출력합니다.
    #     print(f"전체 파라미터 수: {total_params}")
    #     print(f"학습 가능한 파라미터 수: {trainable_params}")

    # loss_fn: torch.nn.Module = get_loss_fn(config.loss)() # type: ignore
    # kwargs["compute_loss_func"] = lambda model, batch, *_, **__: loss_fn(model, batch)

    return base_trainer(**kwargs)

class TrainConfig(TaskConfig):

    model_name: str 
    """모델이 저장될 이름, 학습 결과는 이 path에 저장됨"""
    base_trainer: str | CallConfig = "Trainer"
    
    reader: ReaderConfig
    format: str | CallConfig
    dataloader: DataLoaderConfig
    tokenizer: TokenizerConfig | None = None
    
    model: ModelConfig
    """학습을 진행할 메인 모델"""

    ref_model: ModelConfig | None = None
    """DPO 등의 preference learning 과정에서 사용할 reference model. 없을 경우 Trainer default 사용"""
    reward_model: ModelConfig | None = None
    """PPO 등의 learning 과정에서 reward를 계산할 때 사용할 모델"""


    loss: CallConfig | None = None
    optimizer: CallConfig | None = None
    scheduler: CallConfig | None = None

    training_arguments: BaseConfig = Field(default_factory=lambda: BaseConfig())
    """hf trainer config. check https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/trainer#transformers.TrainingArguments"""


    def __call__(self):
        """Config를 사용하여 모델을 학습합니다."""
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        trainer = create_trainer(self)
        if self.tokenizer is None:
            self.tokenizer = TokenizerConfig(path=self.model.path)
        if is_rank_zero():
            print("start training...")
            print(self.training_arguments)
            print(self.model)
            self.tokenizer().save_pretrained(save_dir)
        trainer.train()
        print(f"{rank()}/{world_size()}: training finished")
        if is_rank_zero():
            self.model.path = save_dir
            self.dump(os.path.join(save_dir, "config.yaml"))
        
        if not getattr(self.training_arguments, "deepspeed", {}):
            # deepspeed 환경에서 이거 부르면 영원히 끝나지 않는다. 대신 convert_checkpoint를 부를 것
            trainer.save_model(save_dir)
        else:
            if is_rank_zero():
                from src.utils import convert_checkpoint
                convert_checkpoint(save_dir)
    
    @property
    def save_dir(self):
        return os.path.join(MODEL_SAVE_DIR, self.model_name)