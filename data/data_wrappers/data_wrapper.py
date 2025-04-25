import os 
from dataclasses import dataclass
from typing import Optional, Dict, List
from torch import Tensor
from transformers import PreTrainedTokenizer
import logging 
import json
import hydra
from data.datasets.gcd_mm_dataset import GCDMM
from data.patterns.pattern_converter import NNSewingPattern
log = logging.getLogger(__name__)


@dataclass 
class DataWrapperConfig:
    split_file: str
    output_dir: str
    

class DataWrapper:
    def __init__(
        self, 
        dataset,
        split_file: str,
        output_dir: str,
        collate_fn: str,
        ):
        self.dataset = dataset
        self.split_file = split_file
        self.output_dir = output_dir
        self.collate_fn = hydra.utils.get_method(collate_fn)
        self.training: Optional[GCDMM] = None
        self.validation: Optional[GCDMM] = None
        self.test: Optional[GCDMM] = None
        self.load_split()
    
    def set_token_indices(self, token2idx: Dict[str, int]):
        self.training.set_token_indices(token2idx)
        self.validation.set_token_indices(token2idx)
        if self.test:
            self.test.set_token_indices(token2idx)
    
    def get_all_token_names(self):
        return self.training.get_all_token_names()
    
    def get_mode_names(self):
        return self.training.get_mode_names()
    
    def decode(self, output_ids: Tensor, tokenizer: PreTrainedTokenizer):
        return self.training.decode(output_ids, tokenizer)
    def evaluate_patterns(self, pred_patterns: List[NNSewingPattern], gt_patterns: List[NNSewingPattern]):
        return self.training.evaluate_patterns(pred_patterns, gt_patterns)
    
    @property
    def panel_edge_type_indices(self):
        return self.training.panel_edge_type_indices
    
    @property
    def gt_stats(self):
        return self.training.gt_stats
    
    def load_split(self):
        if self.split_file is not None and os.path.exists(self.split_file):
            log.info('Loading data split from {}'.format(self.split_file))
            with open(self.split_file, 'r') as f_json:
                split_dict = json.load(f_json)
        else:
            log.error('No valid split file provided. Please provide a split file to load the data split.')
            raise ValueError('No valid split file provided. Please provide a split file to load the data split.')
        self.training = hydra.utils.instantiate(
            self.dataset, 
            load_by_dataname=split_dict["train"],
        )
        
        self.validation = hydra.utils.instantiate(
            self.dataset, 
            load_by_dataname=split_dict["validation"],
        )
        if "test" in split_dict:
            self.test = hydra.utils.instantiate(
                self.dataset, 
                load_by_dataname=split_dict["test"],
            )
        else:
            self.test = None
            

        log.info('Dataset split: {} / {} / {}'.format(
            len(self.training) if self.training else None, 
            len(self.validation) if self.validation else None, 
            len(self.test) if self.test else None))
            
        return self.training, self.validation, self.test
    