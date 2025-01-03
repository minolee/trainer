from __future__ import annotations

from src.base import BaseConfig, CallConfig

from src.data.reader import ReaderConfig
from src.data.dataset import DatasetConfig
from src.data.dataloader import DataLoaderConfig
from src.data import DataModule
from src.tokenizer import TokenizerConfig
from src.model import ModelConfig
from src.task.train import TrainConfig
from src.env import MODEL_SAVE_DIR
import transformers
import torch
import os
from pydantic import Field
from datetime import datetime
__all__ = ["InferenceConfig"]

class InferenceConfig(BaseConfig):

    pretrained_model: str | None = None
    """학습한 모델의 경로 또는 huggingface의 pretrained model card"""
    model: ModelConfig | None = None
    """pretrained_model이 없는 경우 model config"""

    tokenizer: TokenizerConfig | None = None
    """pretrained_model이 없는 경우 tokenizer config"""

    experiment_name: str = Field(default_factory=lambda: datetime.strftime(datetime.now(), "%Y%M%d_%H%m%s"))
    # data configs
    reader: ReaderConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig

    # generation configs
    # generation이 아니라면? 일단은 그냥 놔두고 나중에 task 분리 필요할 듯?
    generation_config: GenerationConfig = Field(default_factory=lambda: GenerationConfig())
    logit_processors: list[str | CallConfig] = Field(default_factory=list)
    out_format: str | None = None

    # deepspeed_config: DeepSpeedConfig | None = None
    def __call__(self):
        # 얘는 call하면 뭘 return해야 되냐..?
        # 일단 setup까지 하는 것은 확정인데
        if self.pretrained_model:
            if os.path.exists((path:=os.path.join(MODEL_SAVE_DIR, self.pretrained_model, "config.yaml"))):
                c = TrainConfig.load(path)
                model_config = c.model
                tokenizer_config = c.tokenizer
            else:
                model_config = ModelConfig(path=self.pretrained_model)
                tokenizer_config = TokenizerConfig(path=self.pretrained_model)
        else:
            assert self.model and self.tokenizer
            model_config = self.model
            tokenizer_config = self.tokenizer
        self.dataloader.padding_side = "left" # force left padding, since position_id is given manually
        datamodule = DataModule(
            self.reader,
            self.dataset,
            self.dataloader,
            tokenizer_config
        )
        tokenizer = datamodule.tokenizer
        model: transformers.GenerationMixin | torch.nn.Module = model_config()
        model.eval()

        datamodule.prepare_data(["test"])
        datamodule.setup(["test"])
        datamodule.info()
        result = []
        generation_config = self.generation_config()
        for batch in datamodule["test"]:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            generated = model.generate(
                batch["input_ids"], 
                generation_config, 
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"]
            )

            inp_size = batch["input_ids"].shape[-1]
            
            for gen in generated[:, inp_size:]:
                result.append(tokenizer.decode(gen, skip_special_tokens=True))
        
        print(result)

class GenerationConfig(BaseConfig):
    """Huggingface generation config를 따름. 정의되지 않은 값은 모델의 generation config를 가져온다.
    
    https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig 참고"""
    # 굳이 모든 field를 가져와서 정의해 둬야 하나? 그냥 extra allow 되어 있으니까 빈 class로 놔둬도 되는 것 아닐까?
    # max_length: int | None = None
    # max_new_tokens: int | None = None
    # min_length: int | None = None
    # min_new_tokens: int | None = None
    # early_stopping: bool | None = None
    # max_time: float | None = None
    # stop_strings: str | list[str] | None = None

    # do_sample: bool | None = None
    # num_beams: int | None = None
    # num_beam_groups: int | None = None
    # penalty_alpha: float | None = None
    # dola_layers: str | list[int] | None = None

    # use_cache: bool | None = None
    # cache_implementation: str | None = None
    # cache_config: dict | None = None
    # return_legacy_cache: bool | None = None

    # temperature: float | None = None
    # top_k: int | None = None
    # top_p: float | None = None
    # min_p: float | None = None
    def __call__(self):
        return transformers.GenerationConfig(**self.model_dump())