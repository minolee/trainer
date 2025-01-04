# BaseMessage + prompt to torch dataset
from __future__ import annotations
import torch
from torch.utils.data import Dataset as D
from transformers import PreTrainedTokenizer
from src.base import create_get_fn, Speaker
from .prompt import PromptTemplate
from src.base import BaseMessage, PreferenceMessage, DataElem
from typing import Iterable
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
        """
        raw data를 가공하여 dataset으로 가공
        """
        self.tokenized = [self.process_elem(x) for x in self.raw_data]


    def __len__(self):
        return len(self.tokenized)
    
    def __getitem__(self, idx):
        return self.tokenized[idx]

    def process_elem(self, elem: DataElem) -> dict[str, torch.Tensor]:
        """
        DataElem을 받아서 모델의 입력으로 가공하는 함수. subclass에서 이 함수를 구현하면 됨.

        :param elem: training / inference 단위 data
        :type elem: DataElem
        :return: 모델의 입력으로 들어갈 tensor. key는 모델의 forward input key와 일치해야 함.
        :rtype: dict[str, torch.Tensor]
        """
        raise NotImplementedError

    def merge_system_prompt(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        # system message should be at the beginning
        if messages[0].speaker == Speaker.SYSTEM:
            messages[0].message = self.prompt.system_prompt + "\n" + messages[0].message
        else:
            messages.insert(0, BaseMessage(speaker=Speaker.SYSTEM, message=self.prompt.system_prompt))
        return messages

get_dataset = create_get_fn(__name__, type_hint=BaseDataset)


class SFTDataset(BaseDataset):
    """
    read from KT style jsonl file, no passage used
    used for sft model
    """
    def process_elem(self, elem: DataElem) -> dict[str, torch.Tensor]:
        messages = elem.elem
        messages = self.merge_system_prompt(messages)
        # 학습 split별로 다른 처리 진행
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
            return {}
        return d


class InferenceDataset(BaseDataset):
    def process_elem(self, elem: DataElem) -> dict[str, torch.Tensor]:
        messages = elem.elem
        messages = self.merge_system_prompt(messages)
        input_ids = []
        attention_mask = []
        for message in messages:
            tok = self.tokenizer(self.prompt.wrap(message), return_tensors="pt")
            input_ids.append(tok["input_ids"].squeeze()) # type: ignore
            attention_mask.append(tok["attention_mask"].squeeze()) # type: ignore
        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
        attention_mask = torch.cat(attention_mask, dim=0)
        if input_ids.shape[0] > self.max_length: 
            # inference 과정에서는 skip하면 망한다
            return {}
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

class PreferenceDataset(BaseDataset):
    def process_elem(self, elem: DataElem) -> dict[str, torch.Tensor]:
        assert isinstance(elem.elem[-1], PreferenceMessage)
        input_ids = []
        attention_mask = []
        for message in elem.elem[:-1]: # last message is used as target
            tok = self.tokenizer(self.prompt.wrap(message), return_tensors="pt")
            input_ids.append(tok["input_ids"].squeeze()) # type: ignore
            attention_mask.append(tok["attention_mask"].squeeze()) # type: ignore
        inference_header_tok = self.tokenizer(self.prompt.inference_header(elem.elem[-1].speaker), return_tensors="pt")
        input_ids.append(inference_header_tok["input_ids"].squeeze())
        attention_mask.append(inference_header_tok["attention_mask"].squeeze())
        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
        attention_mask = torch.cat(attention_mask, dim=0)
        if input_ids.shape[0] > self.max_length: 
            # inference 과정에서는 skip하면 망한다
            return {}
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

class PackingDataset(D):
    ...