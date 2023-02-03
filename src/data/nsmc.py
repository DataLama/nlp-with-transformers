import os
import logging
from omegaconf import OmegaConf
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, load_dataset, load_dataset_builder, ClassLabel
from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorWithPadding,
    default_data_collator
)

from .base import TransformerDataModule

def convert_to_features(
    examples: Any, 
    _, 
    tokenizer, 
    padding,
    truncation,
    max_length,
    return_length
):
    tokenized_inputs = tokenizer(
        text = examples['document'],
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_length=return_length
    )
    tokenized_inputs['label'] = examples['label']
    return tokenized_inputs


class NSMCDataModule(TransformerDataModule):
    """TransformerDataModule for `NSMC` Dataset
    
    Style:
        - Simple Text Classification.

    """
    def __init__(
        self, 
        *args, 
        label_list:List[str]=[],
        return_length:bool=False,
        **kwargs
    ) -> None:
        """
        Args:
            * label_list (List[str]): List of labels
            * return_length (bool): Whether to return length of input_ids.
        """
        super().__init__(*args, **kwargs)
        self._label_list = label_list        
        self._classlabel = None
        self.return_length = return_length

        # get class label from dataset_build_config or label_list
        if self.dataset_name is not None:
            build_config = load_dataset_builder(self.dataset_name)
            for _, v in build_config.info.features.items():
                if isinstance(v, ClassLabel):
                    self._classlabel = v
            if (self._classlabel is None) and (len(self._label_list)==0):
                raise ValueError(f"{self.dataset_name} has no ClassLabel in dataset_build_config. Please pass the label_list.")
        
        # model input columns
        self._input_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
        if self.return_length:
            self._input_columns.append("length")
            
    def get_collate_fn(self, fp16=False) -> Optional[Callable]:
        """Get collate_fn for huggingface Trainer."""
        if fp16:
            return DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        else:
            return DataCollatorWithPadding(self.tokenizer)
        
    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        """Process for text classification"""
        # filter the empty string
        dataset = dataset.filter(lambda x: len(x['document']) > 0)

        # convert to features
        partial_convert_to_features = partial(
            convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_length=self.return_length,
        )
        dataset = dataset.map(
            partial_convert_to_features,
            batched=True,
            with_indices=True,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=self.load_from_cache_file,
        )
        # transform features for model inputs.
        cols_to_keep = [x for x in self._input_columns if x in dataset["train"].features]
        dataset.set_format('torch', columns=cols_to_keep)
        return dataset
    
    @property
    def id2label(self) -> List:
        if self._classlabel:
            return self._classlabel.names
        else:
            return self._label_list
    
    @property
    def label2id(self) -> Dict:
        return {l:i for i, l in enumerate(self.id2label)}
    
    @property
    def num_classes(self) -> int:
        return len(self.id2label)
