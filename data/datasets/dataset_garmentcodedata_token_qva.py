import numpy as np
from scipy.sparse import csr_matrix
import os
from pathlib import Path
import shutil
import pickle 
from collections import defaultdict
import sys
import igl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.spatial.transform import Rotation as scipy_rot
from typing import List, Dict, Tuple, Union, Optional, Literal
from transformers import CLIPImageProcessor, PreTrainedTokenizer
import cv2
import random
import json
# My
from data.garment_tokenizers.default_garment_tokenizer import GarmentTokenizer
import data.datasets.garmentcodedata.transforms as transforms
from data.datasets.garmentcodedata.pattern_converter import NNSewingPattern, InvalidPatternDefError
from data.datasets.garmentcodedata.external.customconfig import Properties
from data.datasets.garmentcodedata.panel_classes import PanelClasses
from data.datasets.utils import (SHORT_QUESTION_LIST, 
                                 ANSWER_LIST, 
                                 DEFAULT_PLACEHOLDER_TOKEN, 
                                 DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SPECULATIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SHORT_QUESTION_WITH_TEXT_LIST,
                                 EDITING_QUESTION_LIST
                                 )
from models.llava import conversation as conversation_lib
from data.transforms import ResizeLongestSide

## incorperating the changes from maria's three dataset classes into a new
## dataset class. this also includes features from sewformer, for interoperability
class GarmentCodeDatasetQVA(Dataset):  
    def __init__(
        self, 
        root_dir: str, 
        editing_dir: str, 
        caption_dir: str, 
        sampling_rate: List[int],
        vision_tower: str,
        image_size: int, 
        editing_flip_prob: float,
        load_by_dataname: List[Tuple[str, str, str]],
        garment_tokenizer: GarmentTokenizer,
        panel_classification: Optional[str]=None,
        inference: bool = False): 

        self.editing_dir = editing_dir
        self.editing_flip_prob = editing_flip_prob
        self.caption_dir = caption_dir
        self.sampling_rate=sampling_rate
        self.panel_classifier = None
        self.panel_classification = panel_classification
        
        #################################
        # init from the basedataset class
        self.root_path = Path(root_dir)
        self.config = {}
        self.config['class'] = self.__class__.__name__


        self.datapoints_names = []
        self.dataset_start_ids = [] # (folder, start_id) tuples -- ordered by start id
        self.panel_classes = []
        self.dict_panel_classes = defaultdict(int)
        self.garment_tokenizer = garment_tokenizer
        self.inference = inference
        
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.vision_tower=vision_tower
        self.image_size = image_size    
        self.transform = ResizeLongestSide(image_size)
        self.short_question_list = SHORT_QUESTION_LIST
        self.descriptive_text_question_list = DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST
        self.speculative_text_question_list = SPECULATIVE_TEXT_SHORT_QUESTION_LIST
        self.text_image_question_list = SHORT_QUESTION_WITH_TEXT_LIST
        self.editing_question_list = EDITING_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.information = defaultdict(dict)  # TODO REMOVE Needed? Should be part of experiment object/initial config? I don't see it used at all 
        ## collect information about data folders: 

        if isinstance(load_by_dataname, str) and load_by_dataname.endswith('.txt'):
            with open(load_by_dataname, 'r') as f:
                load_by_dataname = [line.strip() for line in f.readlines()]
        self.datapoints_names = load_by_dataname
        # Wrong Don't use
        self.dataset_start_ids = [(dn, i) for i, dn in enumerate(load_by_dataname)]
        
           
                
        if panel_classification is not None:
            self.panel_classifier = PanelClasses(classes_file=panel_classification)
            self.panel_classes = self.panel_classifier.classes
        else:
            self.panel_classes = list(sorted(self.dict_panel_classes, key=self.dict_panel_classes.get, reverse=True))
            self.panel_classifier = PanelClasses(self.panel_classes)

        print("The panel classes in this dataset are :", self.panel_classes)
        

        self.gt_cached = {}
        self.gt_caching = True

        # Use default tensor transform + the ones from input
        self.transforms = [transforms.SampleToTensor()]

        ########################################
        # end of init from the basedataset class
        
        # self._drop_cache()
        self.gt_cached = {}


        self.ready = True


    # my method for colleting all the panel classes in the dataset 
    # might need to be seperated into a different class to be evaluated
    # independent of every run of the dataset
    def _collect_panel_classes(self, dirs, root_path):
        all_classes = defaultdict(int)

        for dir in dirs:
            dir_path = root_path / dir
            _, _, files = next(os.walk(dir_path))

            spec = [f for f in files if 'specification.json' in f]
            if len(spec) == 0:
                continue

            # TODO use it? Since it's loaded
            pattern = NNSewingPattern(dir_path / spec[0])
            panel_names = pattern.pattern['panels'].keys() 
            for p in panel_names:
                all_classes[p] += 1

        return all_classes

    # added from maria
    def _read_pattern(self, datapoint_name, folder_elements, 
                      pad_panels_to_len=None, pad_panel_num=None, pad_stitches_num=None,
                      with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Read given pattern in tensor representation from file"""
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('GarmentBaseDataset::Error::*specification.json not found for {}'.format(datapoint_name))
        pattern = NNSewingPattern(
            self.root_path / datapoint_name / spec_list[0], 
            panel_classifier=self.panel_classifier, 
            template_name=self.template_name(datapoint_name))
        return pattern.pattern_as_tensors(
            pad_panels_to_len, pad_panels_num=pad_panel_num, pad_stitches_num=pad_stitches_num,
            with_placement=with_placement, with_stitches=with_stitches, 
            with_stitch_tags=with_stitch_tags)

    # added from maria 
    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)  

    def _fetch_pattern(self, idx):
        """Fetch the pattern and image for the given index"""
        _, garment_names, garment_spec = self.datapoints_names[idx]
        gt_pattern = NNSewingPattern(garment_spec)
        gt_pattern.name = "_".join(garment_names)
        return gt_pattern
    
    def _parepare_image(self, image_paths):
        """Fetch the image for the given index"""
        image_path = random.choice(image_paths)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]  # preprocess image for clip
        return image_clip, image_path
    
    def get_mode_names(self):
        return ['image', 'description','occasion', 'text_image', 'editing']

    # added from maria 
    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]
        data_name = datapoint_name.split('/')[-1]
        image_paths = [
            os.path.join(self.root_path, datapoint_name, f'{data_name}_render_back.png'),
            os.path.join(self.root_path, datapoint_name, f'{data_name}_render_front.png')
        ]
        if data_name in self.gt_cached:
            gt_pattern, pattern_dict, edited_pattern, edited_pattern_dict, editing_captions, captions = self.gt_cached[data_name]
        else:
            spec_file = os.path.join(self.root_path, datapoint_name, f'{data_name}_specification_shifted.json')
            gt_pattern = NNSewingPattern(spec_file, panel_classifier=self.panel_classifier, template_name=self.template_name(datapoint_name))
            gt_pattern.name = data_name
            pattern_dict = self.garment_tokenizer.encode(gt_pattern)
            
            editing_spec_file = os.path.join(self.editing_dir, data_name, f'edited_specification.json')
            editing_caption_json = os.path.join(self.editing_dir, data_name, f'editing_caption.json')
            if (not os.path.exists(editing_spec_file)) or (not os.path.exists(editing_caption_json)):
                edited_pattern = None
                edited_pattern_dict = None
                editing_captions = None
            else:
                edited_pattern = NNSewingPattern(editing_spec_file, panel_classifier=self.panel_classifier, template_name=self.template_name(datapoint_name))
                edited_pattern.name = data_name
                edited_pattern_dict = self.garment_tokenizer.encode(edited_pattern)
                editing_captions = json.load(open(editing_caption_json, 'r'))
            
            caption_json = os.path.join(self.caption_dir, data_name, f'captions.json')
            if os.path.exists(caption_json):
                captions = json.load(open(caption_json, 'r'))
            else:
                captions = None
            
            self.gt_cached[data_name] = (gt_pattern, pattern_dict, edited_pattern, edited_pattern_dict, editing_captions, captions)
            
            
        image_clip = torch.zeros((3, 224, 224))
        image_path = ''
        sample_type = np.random.choice(5, p=self.sampling_rate)
        if sample_type == 4 and edited_pattern is None:
            sample_type = 0  # no editing if there is no edited pattern
        if sample_type in [1, 2, 3] and captions is None:
            sample_type = 0  # no text if there is no caption
        if sample_type == 0:
            # image_only
            image_clip, image_path = self._parepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.short_question_list)
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == 1:
            # descriptive text_only
            descriptive_text = captions['description']
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.descriptive_text_question_list).format(sent=descriptive_text)
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == 2:
            # speculative text_only
            speculative_text = captions['occasion']
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.speculative_text_question_list).format(sent=speculative_text)
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == 3:
            # image_text
            descriptive_text = captions['description']
            image_clip, image_path = self._parepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.text_image_question_list).format(sent=descriptive_text)
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append(answer_template)
            out_pattern_dict = pattern_dict
            question_pattern_dict = {}
            out_pattern = [gt_pattern]
        elif sample_type == 4:
            # garment_editing
            if random.random() > self.editing_flip_prob:
                before_pattern_dict = pattern_dict
                after_pattern_dict = edited_pattern_dict
                editing_text = editing_captions['editing_description_forward']
                gt_pattern.name = "before_" + gt_pattern.name
                edited_pattern.name = "after_" + edited_pattern.name
                out_pattern = [gt_pattern, edited_pattern]
            else:
                before_pattern_dict = edited_pattern_dict
                after_pattern_dict = pattern_dict
                editing_text = editing_captions['editing_description_reverse']
                edited_pattern.name = "before_" + edited_pattern.name
                gt_pattern.name = "after_" + gt_pattern.name
                out_pattern = [edited_pattern, gt_pattern]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.editing_question_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN, sent=editing_text)
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append(answer_template)
            out_pattern_dict = {k: before_pattern_dict[k] + after_pattern_dict[k] for k in before_pattern_dict.keys()}
            question_pattern_dict = before_pattern_dict

        conversations = []
        question_only_convs = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], "")
            question_only_convs.append(conv.get_prompt())
            i += 1

        return (
            out_pattern_dict,
            question_pattern_dict,
            image_path,
            image_clip,
            conversations,
            question_only_convs,
            questions,
            out_pattern,
            sample_type,
            self.inference
        ) 
        
    def evaluate_patterns(self, pred_patterns: List[NNSewingPattern], gt_patterns: List[NNSewingPattern]):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)
    
    def _as_vertices(self, outlines, num_edges):
        
        
        vertices = np.zeros(outlines.shape)
        
        valid_edges = outlines[num_edges > 0]
        valid_vertices = outlines[num_edges > 0]

        #vertices themselves
        for i in range(valid_edges.shape[1]):
            valid_vertices[:,i,:2] += valid_edges[:, i, :2]

        # for control points 
        for i in range(valid_edges.shape[0]):
            for j in range(valid_edges.shape[1]):
                if valid_edges[i, j, 2:4].sum() > 0:
                    valid_vertices[i,j,2:4] = NNSewingPattern.control_to_abs_coord(valid_vertices[i,j,:2], valid_vertices[i,j+1,:2], valid_edges[i,j,2:4])
                if valid_edges[i, j, 4:6].sum() > 0:
                    valid_vertices[i,j,4:6] = NNSewingPattern.control_to_abs_coord(valid_vertices[i,j,:2], valid_vertices[i+1,j,:2], valid_edges[i,j,4:6])
        
        vertices[num_edges > 0] = valid_vertices
        
        return vertices

    
    def get_item_infos(self, index):
        return self.datapoints_names[index]
    
    def decode(self, output_ids: torch.Tensor, tokenizer: PreTrainedTokenizer):
        """Decode output ids to text"""
        return self.garment_tokenizer.decode(output_ids, tokenizer)
    
    def split_from_dict(self, split_dict):
        """
            Reproduce the data split in the provided dictionary: 
            the elements of the currect dataset should play the same role as in provided dict
        """
        train_ids, valid_ids, test_ids = [], [], []
        
        training_datanames = split_dict['train']
        valid_datanames = split_dict['validation']
        test_datanames = split_dict['test']

        for idx in range(len(self.datapoints_names)):
            if self.datapoints_names[idx] in training_datanames:  # usually the largest, so check first
                train_ids.append(idx)
            elif self.datapoints_names[idx] in test_datanames:
                test_ids.append(idx)
            elif self.datapoints_names[idx] in valid_datanames:
                valid_ids.append(idx)
            else:
                continue
            
            if idx % 1000 == 0:
                print(f"progress {idx}, #Train_Ids={len(train_ids)}, #Valid_Ids={len(valid_ids)}, #Test_Ids={len(test_ids)}")
        
        return (
            [self.datapoints_names[i] for i in train_ids], 
            [self.datapoints_names[i] for i in valid_ids],
            [self.datapoints_names[i] for i in test_ids] if len(test_ids) > 0 else None
        )
    