"""데이터를 총괄하는 모듈 - 아마 안쓸듯?"""
from .reader import ReaderConfig, BaseDataset
from .dataset import DatasetConfig
from .dataloader import DataLoaderConfig
from src.tokenizer import TokenizerConfig
from src.utils import rank_zero_only
from torch.utils.data import Dataset, ConcatDataset
from transformers import PreTrainedTokenizer
__all__ = ['DataModule']


class DataModule:
    # LightningDataModule이 dataset이랑 dataloader를 합쳐놓은 것인데 dataset이랑 dataloader가 지금 분리돼있어서 어떻게 구현해야 좋을지 생각해 봐야 함

    def __init__(
        self, 
        reader_config: ReaderConfig,
        dataset_config: DatasetConfig,
        tokenizer_config: TokenizerConfig
    ):
        super().__init__()
        self.reader_config = reader_config
        self.dataset_config = dataset_config
        self.prepared: dict[str, list[BaseDataset]] = {} # before setup
        # self.processed: dict[str, Dataset] = {}
        
        self.tokenizer: PreTrainedTokenizer = tokenizer_config() # type: ignore
    
    def prepare_data(self, stage: str | list[str]):
        print("Preparing data")
        self.reader_config()
        self.reader_config.info()
        stages = [stage] if isinstance(stage, str) else stage
        for stage in stages:
            self.prepared[stage] = []
            target_data = self.reader_config[stage]
            dataset: SFTDataset = self.dataset_config(stage, target_data, self.tokenizer) # type: ignore
            
            self.prepared[stage].append(dataset)
            
    def setup(self, stage : str | list[str]):
        # stage: fit, validate, test, predict
        stages = [stage] if isinstance(stage, str) else stage
        
        for stage in stages:
            for dataset in self.prepared[stage]:
                dataset.setup()
    
    @rank_zero_only
    def info(self, stage: str | list[str] | None = None):
        if stage is None:
            stage = ["train", "dev", "test"]
        stages = [stage] if isinstance(stage, str) else stage
        for stage in stages:
            if stage in self.prepared:
                print(f"Stage: {stage}")
                for dataset in self.prepared[stage]:
                    print(f"Dataset: {dataset.__class__.__name__}")
                    print(f"Number of data: {len(dataset)}")

    def __getitem__(self, key: str):
        return ConcatDataset(self.prepared[key])
    