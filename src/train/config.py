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

import transformers
import trl

get_trainer = create_get_fn(transformers, trl) # type: ignore


class TrainConfig(BaseConfig):

    model_name: str # 모델이 저장될 이름, 이 path에 저장됨
    base_trainer: str | CallConfig = "Trainer"
    
    reader_config: ReaderConfig
    dataset_config: DatasetConfig
    dataloader_config: DataLoaderConfig
    tokenizer_config: TokenizerConfig
    model_load_config: ModelConfig # model_config가 안되는거 실화냐

    loss_config: CallConfig
    optimizer_config: CallConfig
    scheduler_config: CallConfig

    trainer_config: dict = Field(default_factory=dict) 
    """used for hf trainer config. check https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/trainer#transformers.TrainingArguments"""


    def __call__(self) -> transformers.Trainer:
        datamodule = DataModule(
            self.reader_config,
            self.dataset_config,
            self.dataloader_config,
            self.tokenizer_config
        )
        base_trainer: type[transformers.Trainer] = get_trainer(self.base_trainer)
        datamodule.prepare_data(["train", "dev"])
        datamodule.setup(["train", "dev"]) # type: ignore

        model = self.model_load_config()
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
        loss_fn: torch.nn.Module = get_loss_fn(self.loss_config)() # type: ignore
        tc = self.trainer_config
        name = self.model_name
        class _Trainer(base_trainer):
            def __init__(self):
                super().__init__(model, transformers.TrainingArguments(f"{MODEL_SAVE_DIR}/{name}", **tc))
            
            def get_train_dataloader(self):
                return datamodule.train_dataloader()
            
            def get_eval_dataloader(self):
                return datamodule.val_dataloader()
            
            def get_test_dataloader(self):
                return datamodule.test_dataloader()

            def compute_loss(self, model, batch, *_, **__):
                inp_tensor = {k: v.to(model.device) for k, v in batch.items()}
                label = inp_tensor.pop("label")
                logits = self.model(**inp_tensor).logits
                loss = loss_fn(logits.view(-1, logits.shape[-1]), label.view(-1))
                return loss
        
        return _Trainer()