# read raw file and split into train-dev-test
from .config import ReaderConfig, ReaderElem
from src.base import BaseMessage
from .reader_fn import get_reader_fn
from typing import Iterable
__all__ = ['Reader']

# type definition
Dialogue = list[BaseMessage]
Corpus = list[Dialogue]

class Reader:
    def __init__(self, config: ReaderConfig):
        self.config = config
        self.corpus: dict[str, list[tuple[Corpus, ReaderElem]]] = {} # store raw data
    
    def load_corpus(self):
        # set default value for reader_fn and prompt
        for source in self.config.sources:
            if source.reader_fn is None:
                source.reader_fn = self.config.default_reader_fn
            if source.prompt is None:
                source.prompt = self.config.default_prompt
            if source.dataset is None:
                source.dataset = self.config.default_dataset
        # load corpus
        corpus = {
            "train": [],
            "dev": [],
            "test": []
        }
        for source in self.config.sources:
            # raw data에 dataset loading function을 함께 묶어놔야 하는데..?
            # 와 진짜 못생겼다 어떡하냐
            train, dev, test = source.split(source.read())
            corpus["train"].append((train, source))
            corpus["dev"].append((dev, source))
            corpus["test"].append((test, source))
        self.corpus = corpus

    
    def __getitem__(self, split):
        return self.corpus[split]