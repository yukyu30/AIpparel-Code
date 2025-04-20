import numpy as np
import os 
import torch 
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Union, Callable
import logging 
import time
import random
import json
import hydra
log = logging.getLogger(__name__)
from ..datasets.dataset_garmentcodedata_token_qva import GarmentCodeDatasetQVA

@dataclass 
class GarmentImageDataWrapperConfig:
    split_file: str
    output_dir: str
    

class GarmentImageDataWrapper:
    def __init__(
        self, 
        dataset: GarmentCodeDatasetQVA,
        split_file: str,
        output_dir: str,
        collate_fn: str,
        ):
        self.dataset = dataset
        self.split_file = split_file
        self.output_dir = output_dir
        self.collate_fn = hydra.utils.get_method(collate_fn)
        self.load_split()
        
    def load_split(self, random_seed: Optional[int] = None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset.
            NOTE this function re-initializes torch random number generator!
        """
        
        if random_seed is None:
            random_seed = int(time.time())
        # init for all libs =)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # if file is provided
        if self.split_file is not None and os.path.exists(self.split_file):
            log.info('Loading data split from {}'.format(self.split_file))
            with open(self.split_file, 'r') as f_json:
                split_dict = json.load(f_json)
            training_datapoints, validation_datapoints, test_datapoints = self.dataset.split_from_dict(
                split_dict)
        else:
            log.error('No valid split file provided. Please provide a split file to load the data split.')
        self.training = GarmentCodeDatasetQVA(
            self.dataset.root_path, 
            self.dataset.editing_dir,
            self.dataset.caption_dir,
            self.dataset.sampling_rate,
            self.dataset.vision_tower,
            self.dataset.image_size,
            self.dataset.editing_flip_prob,
            training_datapoints,
            self.dataset.garment_tokenizer,
            panel_classification=self.dataset.panel_classification,
        )
        self.validation = GarmentCodeDatasetQVA(
            self.dataset.root_path, 
            self.dataset.editing_dir,
            self.dataset.caption_dir,
            self.dataset.sampling_rate,
            self.dataset.vision_tower,
            self.dataset.image_size,
            self.dataset.editing_flip_prob,
            validation_datapoints,
            self.dataset.garment_tokenizer,
            panel_classification=self.dataset.panel_classification,
            inference=True
        )
        if test_datapoints is not None:
            self.test = GarmentCodeDatasetQVA(
                self.dataset.root_path, 
                self.dataset.editing_dir,
                self.dataset.caption_dir,
                self.dataset.sampling_rate,
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.editing_flip_prob,
                test_datapoints,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                inference=True
            )
        else:
            self.test = None
            

        log.info('Dataset split: {} / {} / {}'.format(
            len(self.training) if self.training else None, 
            len(self.validation) if self.validation else None, 
            len(self.test) if self.test else None))
            
        self.get_data_lists()
            
        return self.training, self.validation, self.test
    

    def get_data_lists(self):

        if self.split_file is not None and  os.path.exists(self.split_file):
            log.info('Load Dataset split: {} '.format(self.split_file))
        else:
            data_lists = {"train": [], "validation": [], "test": []}
            for name, split in {"train":self.training, "validation": self.validation, "test": self.test}.items():
                for idx in range(len(split)):
                    dataname = split.get_item_infos(idx)
                    data_lists[name].append(dataname)
                    
            save_path = os.path.join(self.output_dir, "data_split.json")
            json.dump(data_lists, open(save_path, "w"), indent=2)
            self.split_file = save_path

            log.info('Save Dataset split: {} '.format(self.split_file))
