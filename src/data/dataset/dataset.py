# BaseMessage + prompt to torch dataset
from __future__ import annotations
import torch
from torch.utils.data import Dataset as D
from transformers import PreTrainedTokenizer
from src.base import create_register_deco, create_get_fn, Speaker
from .prompt import PromptTemplate
from src.base import BaseMessage, DataElem
from typing import Iterable
import sys
__all__ = ["get_dataset", "BaseDataset"]


class BaseDataset(D):
    def __init__(
        self, 
        data: Iterable[DataElem], 
        split: str,
        prompt: PromptTemplate,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        **kwargs
    ):
        self.raw_data = data
        self.split = split
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.tokenized = []
        self.max_length = max_length
        self.train_every_assistant_message = kwargs.get("train_every_assistant_message", False)
    
    def setup(self):
        # setup logic
        raise NotImplementedError

get_dataset = create_get_fn(__name__, type_hint=BaseDataset)


class SFTDataset(BaseDataset):
    """
    read from KT style jsonl file, no passage used
    used for sft model
    """
    def __init__(
        self, 
        data: Iterable[DataElem], 
        split: str,
        prompt: PromptTemplate,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        **kwargs
    ):
        self.raw_data = data
        self.split = split
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.tokenized = []
        self.max_length = max_length
        self.train_every_assistant_message = kwargs.get("train_every_assistant_message", False)
        
    def setup(self):
        for messages in self.raw_data:
            messages = messages.elem
            # system message should be at the beginning
            if messages[0].speaker == Speaker.SYSTEM:
                messages[0].message = self.prompt.system_prompt + "\n" + messages[0].message
            else:
                messages.insert(0, BaseMessage(speaker=Speaker.SYSTEM, message=self.prompt.system_prompt))
            # 학습 split별로 다른 처리 진행
            if self.split in ["train", "dev"]:
                input_ids = []
                attention_mask = []
                loss_mask = []
                for i, message in enumerate(messages):
                    tok = self.tokenizer(self.prompt.wrap(message), return_tensors="pt")
                    input_ids.append(tok["input_ids"].squeeze()) # type: ignore
                    attention_mask.append(tok["attention_mask"].squeeze()) # type: ignore
                    loss_mask.append(torch.tensor([
                        message.speaker.type == "Assistant" and (self.train_every_assistant_message or i == len(messages) - 1)
                    ] * tok["input_ids"].shape[-1])) # type: ignore
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
                if input_ids.shape[0] > self.max_length:
                    continue
                self.tokenized.append(d)
            elif self.split in ["test", "predict"]:
                input_ids = []
                attention_mask = []
                for message in messages[:-1]: # last message is used as target
                    tok = self.tokenizer(self.prompt.wrap(message), return_tensors="pt")
                    input_ids.append(tok["input_ids"].squeeze()) # type: ignore
                    attention_mask.append(tok["attention_mask"].squeeze()) # type: ignore
                inference_header_tok = self.tokenizer(self.prompt.inference_header(messages[-1].speaker), return_tensors="pt")
                input_ids.append(inference_header_tok["input_ids"].squeeze())
                attention_mask.append(inference_header_tok["attention_mask"].squeeze())
                input_ids = torch.cat(input_ids, dim=0)
                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
                attention_mask = torch.cat(attention_mask, dim=0)
                if input_ids.shape[0] > self.max_length: 
                    # inference 과정에서는 skip하면 망한다
                    self.tokenized.append({})
                self.tokenized.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids
                })
            
    def __len__(self):
        return len(self.tokenized)
    
    def __getitem__(self, idx):
        return self.tokenized[idx]


class PreferenceDataset(D):
    ...

class PackingDataset(D):
    ...