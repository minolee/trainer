# BaseMessage + prompt to torch dataset
from __future__ import annotations
import torch
from torch.utils.data import Dataset as D
from transformers import PreTrainedTokenizer
from src.base import create_register_deco, Speaker
from .prompt import get_prompt
from .config import ReaderElem
from src.base import BaseMessage
from typing import Iterable

__all__ = ["get_dataset", "list_dataset", "BaseDataset"]

_dataset: dict[str, type[D]] = {}

dataset = create_register_deco(_dataset)

def get_dataset(name: str) -> type[D]:
    return _dataset[name.lower()]

def list_dataset():
    return _dataset.keys()

@dataset
class BaseDataset(D):
    # read from KT style jsonl file, no passage used
    # used for sft model
    def __init__(
        self, 
        data: Iterable[list[BaseMessage]], 
        split: str,
        config: ReaderElem,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        assert config.prompt
        self.raw_data = data
        self.split = split
        self.tokenizer = tokenizer
        self.prompt = get_prompt(config.prompt)(**(config.prompt_kwargs or {}))
        self.tokenized = []
        self.train_every_assistant_message = kwargs.get("train_every_assistant_message", False)
        
    def setup(self):
        for messages in self.raw_data:
            # system message should be at the beginning
            if messages[0].speaker == Speaker.SYSTEM:
                messages[0].message = self.prompt.system_prompt + "\n" + messages[0].message
            else:
                messages.insert(0, BaseMessage(speaker="system", message=self.prompt.system_prompt))
            # 학습 split별로 다른 처리 진행
            if self.split in ["train", "dev"]:
                input_ids = []
                attention_mask = []
                loss_mask = []
                for i, message in enumerate(messages):
                    tok = self.tokenizer(self.prompt.wrap(message), return_tensors="pt")
                    input_ids.append(tok["input_ids"].squeeze())
                    attention_mask.append(tok["attention_mask"].squeeze())
                    loss_mask.append(torch.tensor([
                        message.speaker.type == "Assistant" and (self.train_every_assistant_message or i == len(messages) - 1)
                    ] * tok["input_ids"].shape[-1]))
                input_ids = torch.cat(input_ids, dim=0)[:-1] # exclude final eot token
                attention_mask = torch.cat(attention_mask, dim=0)[:-1]
                loss_mask = torch.cat(loss_mask, dim=0)[:-1]
                label = torch.cat([input_ids[1:], torch.tensor([self.tokenizer.eos_token_id])], dim=0) # finish with eos token
                label = torch.where(loss_mask, label, -100)
                d = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": label
                }
                if self.split == "dev":
                    del d["loss_mask"]
                self.tokenized.append(d)
            elif self.split == "test":
                input_ids = []
                attention_mask = []
                for message in messages[:-1]: # last message is used as target
                    tok = self.tokenizer(self.prompt.wrap(message), return_tensors="pt")
                    input_ids.append(tok["input_ids"].squeeze())
                    attention_mask.append(tok["attention_mask"].squeeze())
                inference_header_tok = self.tokenizer(self.prompt.inference_header(messages[-1].speaker), return_tensors="pt")
                input_ids.append(inference_header_tok["input_ids"].squeeze())
                attention_mask.append(inference_header_tok["attention_mask"].squeeze())
                input_ids = torch.cat(input_ids, dim=0)
                attention_mask = torch.cat(attention_mask, dim=0)
                self.tokenized.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })
            
    def __len__(self):
        return len(self.tokenized)
    
    def __getitem__(self, idx):
        return self.tokenized[idx]


@dataset
class PreferenceDataset(D):
    ...

@dataset
class PackingDataset(D):
    ...