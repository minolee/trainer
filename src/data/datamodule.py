
from .reader import ReaderConfig, Reader, get_dataset, BaseDataset
from .dataloader import DataLoaderConfig, get_collate_fn
from ..tokenizer import TokenizerConfig, load_tokenizer
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from functools import partial
from transformers import PreTrainedTokenizer
__all__ = ['DataModule']
def create_dataloader(dataset: Dataset, config: DataLoaderConfig, tokenizer: PreTrainedTokenizer):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=partial(
            get_collate_fn(config.collate_fn),
            pad_id = tokenizer.pad_token_id,
            padding_side = config.padding_side or tokenizer.padding_side
        )
    )

class DataModule(LightningDataModule):
    # LightningDataModule이 dataset이랑 dataloader를 합쳐놓은 것인데 dataset이랑 dataloader가 지금 분리돼있어서 어떻게 구현해야 좋을지 생각해 봐야 함

    def __init__(
        self, 
        reader_config: ReaderConfig,
        dataloader_config: DataLoaderConfig,
        tokenizer_config: TokenizerConfig
    ):
        super().__init__()
        self.reader = Reader(reader_config)
        self.dataloader_config = dataloader_config
        self.prepared: dict[str, list[BaseDataset]] = {} # before setup
        self.processed: dict[str, Dataset] = {}
        
        self.tokenizer = load_tokenizer(tokenizer_config)
    
    def prepare_data(self):
        self.reader.load_corpus()
        stages = ["train", "dev", "test"]
        for stage in stages:
            self.prepared[stage] = []
            target_data = self.reader[stage]
            
            for dialogue, config in target_data:
                assert config.dataset
                dataset_cls = get_dataset(config.dataset)
                dataset = dataset_cls(dialogue, stage, config, self.tokenizer)
                self.prepared[stage].append(dataset)
            
    def setup(self, stage):
        # stage: fit, validate, test, predict
        stages = []
        if stage == "fit":
            stages = ["train", "dev"]
        elif stage == "validate":
            stages = ["dev"]
        else:
            stages = ["test"]
        for stage in stages:
            if stage in self.processed: continue
            for dataset in self.prepared[stage]:
                dataset.setup()
            self.processed[stage] = ConcatDataset(self.prepared[stage])
    
    def train_dataloader(self):
        return create_dataloader(self.processed["train"], self.dataloader_config, self.tokenizer) # type: ignore

    def val_dataloader(self):
        return create_dataloader(self.processed["dev"], self.dataloader_config, self.tokenizer) # type: ignore

    def test_dataloader(self):
        return create_dataloader(self.processed["test"], self.dataloader_config, self.tokenizer) # type: ignore