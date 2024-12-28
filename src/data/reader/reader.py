# read raw file and split into train-dev-test
from .config import ReaderConfig, ReaderElem
from src.base import BaseMessage
from .reader_fn import get_reader_fn
from typing import Iterable
__all__ = ['Reader']

# type definition
Dialogue = list[BaseMessage]
Corpus = list[Dialogue]

def load_data(source: ReaderElem):
    # raw data to input prompt
    assert source.reader_fn is not None, f"reader_fn of {source.name or source.source} is not defined"
    # assert source.prompt is not None, f"prompt of {source.name or source.source} is not defined"
    reader_fn = get_reader_fn(source.reader_fn)
    return reader_fn(source.source)

def split_data(data: Iterable[Dialogue], split_ratio: list[int]):
    # split data into train / dev / test
    # split - dialogue - utterance
    split_data: list[Corpus] = [[] for _ in range(len(split_ratio))]
    
    c = 0
    idx = 0
    it = iter(data)
    while True:
        try:
            if c >= split_ratio[idx]:
                c = 0
                idx += 1
                idx %= len(split_ratio)
            line = next(it)
            split_data[idx].append(line)
            c += 1
        except StopIteration:
            break
    return split_data

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
            train, dev, test = split_data(load_data(source), source.split.parse_split_ratio())
            corpus["train"].append((train, source))
            corpus["dev"].append((dev, source))
            corpus["test"].append((test, source))
        self.corpus = corpus

    
    def __getitem__(self, split):
        return self.corpus[split]