import pytest

from src.base import BaseConfig
from src.data.reader import ReaderConfig
from src.data.dataloader import DataLoaderConfig
from src.tokenizer import TokenizerConfig
from src.data.datamodule import DataModule
import torch

class DataTestConfig(BaseConfig):
    reader: ReaderConfig
    loader: DataLoaderConfig
    tokenizer: TokenizerConfig

@pytest.fixture
def base_datamodule():
    config = DataTestConfig.load("test/config/data_load_config.yaml")
    assert len(config.reader.sources) == 1
    dm = DataModule(
        config.reader,
        config.loader,
        config.tokenizer
    )
    dm.prepare_data()
    assert "train" in dm.reader.corpus
    dm.setup("fit")
    return dm

# TODO dialog dataset test

class TestBaseData:
    def test_dataset_setup(self, base_datamodule: DataModule):
        train_dataset = base_datamodule.prepared["train"][0]
        assert len(set(train_dataset[0][k].shape for k in train_dataset[0])) == 1
        assert len(train_dataset) == 1 # type: ignore
        
        msg = base_datamodule.reader.corpus["train"][0][0][0]
        prompt = base_datamodule.prepared["train"][0].prompt

        tokenizer = base_datamodule.tokenizer
        decoded = tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=False)
        for message in msg:
            if message.speaker.type == "System": continue
            assert prompt.wrap(message).rstrip(prompt.eos_token) in decoded
        
    def test_loss_mask(self, base_datamodule: DataModule):
        train_dataset = base_datamodule.prepared["train"][0]
        msg = base_datamodule.reader.corpus["train"][0][0][0]
        prompt = base_datamodule.prepared["train"][0].prompt

        tokenizer = base_datamodule.tokenizer
        mask_applied_label = torch.where(train_dataset[0]["label"] != -100, train_dataset[0]["label"], 0)
        zero_id = tokenizer.convert_ids_to_tokens(0)[0]
        target_decoded = tokenizer.decode(mask_applied_label).lstrip(zero_id)
        # assert torch.all(train_dataset[0]["input_ids"][1:] == train_dataset[0]["label"][:-1])
        for message in msg:
            if message.speaker.type.lower() in ["system", "user"]: continue
            assert message.message in target_decoded
        