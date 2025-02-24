from __future__ import annotations
from src.base import BaseConfig, CallConfig
from src.data import DataConfig
from src.model import ModelConfig
from src.tokenizer import TokenizerConfig
from src.env import MODEL_SAVE_DIR
from src.utils import world_size, is_rank_zero, rank, drop_unused_args, create_get_fn, rank_zero_print
from . import preprocess_args as P
from . import postprocess_trainer as POST
from . import custom as C
from pydantic import Field

import torch
import transformers
import trl
import os
import inspect

get_trainer = create_get_fn(transformers, C, trl, type_hint=transformers.Trainer)
get_optimizer = create_get_fn(torch.optim, type_hint=torch.optim.Optimizer)

def create_trainer(config: TrainConfig):
    """Config를 사용하여 Trainer를 생성합니다. Deepspeed config가 있을 경우 필요한 설정을 추가합니다."""
    
    kwargs = {} # PPO같은 일부 trainer들은 model, arg 순서로 받지 않아서 kwargs로 넘겨줌

    name = config.model_name
    # per_device_train_batch_size = getattr(config.training_arguments, "per_device_train_batch_size", 8)
    # config.dataloader.batch_size = per_device_train_batch_size
    # if hasattr(config.training_arguments, "deepspeed"):
    #     config.training_arguments.deepspeed["train_micro_batch_size_per_gpu"] = per_device_train_batch_size # type: ignore
        # config.training_arguments.load_best_model_at_end = True
    # set trainer and training argument class
    assert isinstance(config.trainer, CallConfig)
    assert isinstance(config.model, ModelConfig)
    if getattr(config.trainer, "report_to", None) == "wandb":
        import wandb
        os.environ["WANDB_PROJECT"] = name
        with open(".env/wandb", encoding="UTF8") as f:  
            wandb.login(key=f.read().strip())

    base_trainer: type[transformers.Trainer] = get_trainer(config.trainer.name)
    argument_cls = inspect.signature(base_trainer.__init__).parameters["args"].annotation
    if not inspect.isclass(argument_cls): # maybe union or optional type
        argument_cls = argument_cls.__args__[0]
    
    config_kwargs = {
        k: getattr(P, k, lambda x: x)(v) for k, v in config.trainer.get_kwargs().items()
    }
    trainer_kwargs = drop_unused_args(argument_cls.__init__, config_kwargs)
    trainer_remainder = {k: v for k, v in config_kwargs.items() if k not in trainer_kwargs}
    train_args: transformers.TrainingArguments = argument_cls(
        f"{MODEL_SAVE_DIR}/{name}", 
        **trainer_kwargs
    ) # deepspeed init here

    rank_zero_print(train_args)
    
    kwargs |= trainer_remainder
    # rank_zero_print(kwargs)
    
    if config.optimizer:
        train_args.set_optimizer(**config.optimizer.model_dump())
    if config.scheduler:
        train_args.set_lr_scheduler(**config.scheduler.model_dump())
    # if config.loss:
    #     kwargs["compute_loss_func"] = get_loss_fn(config.loss)()
    # load model
    model = config.model()
    rank_zero_print(model)
    assert config.tokenizer
    tokenizer = config.tokenizer()
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs["model"] = model
    kwargs["processing_class"] = tokenizer
    kwargs["args"] = train_args
    
    # if config.ref_model:
    #     ref_model = config.ref_model()
    #     ref_model.eval()
    #     kwargs["ref_model"] = ref_model

    # if config.reward_model:
    #     kwargs["reward_model"] = config.reward_model()
    
    # load data
    datamodule = config.data()

    kwargs["train_dataset"] = datamodule["train"]
    kwargs["eval_dataset"] = datamodule.get("dev", None)
    
    
    try:
        val_elem = datamodule["train"][0]
        
    except:
        val_elem = next(iter(datamodule["train"]))
    rank_zero_print(val_elem)
    
    # postprocess trainer arguments
    if hasattr(POST, base_trainer.__name__):
        kwargs = getattr(POST, base_trainer.__name__)(**kwargs)
    # print(tokenizer.apply_chat_template(val_elem))
    
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

class TrainConfig(BaseConfig):

    model_name: str 
    """모델이 저장될 이름, 학습 결과는 이 path에 저장됨"""
    trainer: str | CallConfig = "Trainer"
    
    data: DataConfig
    """학습에 사용할 데이터를 정의"""
    
    model: ModelConfig | str
    """학습을 진행할 메인 모델"""

    tokenizer: TokenizerConfig | None = None
    """모델에 사용할 tokenizer. 없을 경우 model의 tokenizer를 사용"""

    loss: CallConfig | None = None
    optimizer: CallConfig | None = None
    scheduler: CallConfig | None = None


    is_accelerate: bool = False
    """수동 설정 금지"""
    is_deepspeed: bool = False
    """수동 설정 금지"""
    deepspeed_stage: int | None = None
    """수동 설정 금지"""
    # training_arguments: BaseConfig = Field(default_factory=lambda: BaseConfig())
    """hf trainer config. check https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/trainer#transformers.TrainingArguments"""



    def __call__(self):
        """Config를 사용하여 모델을 학습합니다."""
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(self.model, str):
            self.model = ModelConfig(path=self.model)
        if self.tokenizer is None:
            self.tokenizer = TokenizerConfig(path=self.model.path)
        if isinstance(self.trainer, str):
            self.trainer = CallConfig(name=self.trainer)
        trainer = create_trainer(self)
        # print(rank()) # prints True
        if is_rank_zero():
            print("start training...")
            rank_zero_print(self.model)
            self.tokenizer().save_pretrained(save_dir)
        trainer.train()
        print(f"{rank()}/{world_size()}: training finished")
        if is_rank_zero():
            self.model.path = save_dir
        if self.is_deepspeed and self.deepspeed_stage == 3:
            if is_rank_zero():
                from src.utils import convert_checkpoint
                convert_checkpoint(save_dir)
        else:
            # deepspeed 환경에서 이거 부르면 영원히 끝나지 않는다. 대신 convert_checkpoint를 부를 것
            trainer.model.save_pretrained(save_dir)
    
    @property
    def save_dir(self):
        return os.path.join(MODEL_SAVE_DIR, self.model_name)