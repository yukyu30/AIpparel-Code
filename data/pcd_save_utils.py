from typing import List, Dict, Tuple
import os
import shutil
import torch
import numpy as np
from pathlib import Path
from data.pattern_converter import NNSewingPattern, InvalidPatternDefError
from data.panel_classes import PanelClasses
import json

def save_prediction_batch(predictions: Dict[str, torch.Tensor], datanames: List[str], data_folders: List[str], panel_classifier: PanelClasses, 
                          config: Dict, save_to: str, features=None, weights=None, orig_folder_names=False, **kwargs):
    """ 
        Saving predictions on batched from the current dataset
        Saves predicted params of the datapoint to the requested data folder.
        Returns list of paths to files with prediction visualizations
        Assumes that the number of predictions matches the number of provided data names"""

    prediction_imgs = []
    if 'posterior' in predictions.keys():
        del predictions['posterior']
    for idx, (name, folder) in enumerate(zip(datanames, data_folders)):
        # "unbatch" dictionary
        prediction = {}
        pname = os.path.basename(folder) + "__" + os.path.basename(name.replace(".obj", ""))
        tmp_path = os.path.join(save_to, pname, '_predicted_specification.json')
        if os.path.exists(tmp_path):
            continue
        
        print("Progress {}".format(tmp_path))

        for key in predictions:
            prediction[key] = predictions[key][idx]
        if "pcds" in kwargs:
            prediction["input"] = kwargs["pcds"][idx]
        if "panel_shape" in kwargs:
            prediction["panel_l2"] = kwargs["panel_shape"][idx]
        
        pattern = _pred_to_pattern(panel_classifier, config, prediction, pname)

        try: 
            tag = f'_predicted_{prediction["panel_l2"]}_' if "panel_l2" in prediction else f"_predicted_"
            final_dir = pattern.serialize(save_to, to_subfolder=True, tag=tag)
        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(folder, e))
            continue
        final_file = pattern.name + '_predicted__pattern.png'
        prediction_imgs.append(Path(final_dir) / final_file)
        # save input pcd
        np.savez(os.path.join(final_dir, "input.npy"), points=prediction["input"][..., :3], normals=prediction["input"][..., 3:])
        shutil.copy2(name, str(final_dir))
        # shutil.copy2(name.replace(".png", "cam_pos.json"), str(final_dir))
        shutil.copy2(os.path.join(folder, "static", "spec_config.json"), str(final_dir))

        # copy originals for comparison
        data_prop_file = os.path.join(folder, "data_props.json")
        if os.path.exists(data_prop_file):
            shutil.copy2(data_prop_file, str(final_dir))
    return prediction_imgs

def _pred_to_pattern(panel_classifier: PanelClasses, config, prediction, dataname, return_stitches=False)-> NNSewingPattern:
    """Convert given predicted value to pattern object
    """
    # undo standardization  (outside of generinc conversion function due to custom std structure)
    gt_shifts = config['standardize']['gt_shift']
    gt_scales = config['standardize']['gt_scale']

    for key in gt_shifts:
        if key == 'stitch_tags':  
            # ignore stitch tags update if explicit tags were not used
            continue
        
        pred_numpy = prediction[key].detach().cpu().numpy()
        if key == 'outlines' and len(pred_numpy.shape) == 2: 
            pred_numpy = pred_numpy.reshape(config["max_pattern_len"], config["max_panel_len"], 4)

        prediction[key] = pred_numpy * gt_scales[key] + gt_shifts[key]

    # recover stitches
    if 'stitches' in prediction:  # if somehow prediction already has an answer
        stitches = prediction['stitches']
    elif 'stitch_tags' in prediction: # stitch tags to stitch list 
        pass
    elif 'edge_cls' in prediction and "edge_similarity" in prediction:
        stitches = prediction_to_stitches(prediction['edge_cls'], prediction['edge_similarity'], return_stitches=return_stitches)
    else:
        stitches = None
    
    # Construct the pattern from the data
    pattern = NNSewingPattern(view_ids=False, panel_classifier=panel_classifier)
    pattern.name = dataname

    try: 
        pattern.pattern_from_tensors(
            prediction['outlines'], 
            panel_rotations=prediction['rotations'],
            panel_translations=prediction['translations'], 
            stitches=stitches,
            padded=True)   
    except (RuntimeError, InvalidPatternDefError) as e:
        print('GarmentDetrDataset::Warning::{}: {}'.format(dataname, e))
        pass
    
    return pattern

def prediction_to_stitches(free_mask_logits, similarity_matrix, return_stitches=False):
    free_mask = (torch.sigmoid(free_mask_logits.squeeze(-1)) > 0.5).flatten()
    if not return_stitches:
        simi_matrix = similarity_matrix + similarity_matrix.transpose(0, 1)
        simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(0), -float("inf"))
        simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(-1), 0)
        num_stitches = free_mask.nonzero().shape[0] // 2
    else:
        simi_matrix = similarity_matrix
        num_stitches = simi_matrix.shape[0] // 2
    simi_matrix = torch.triu(simi_matrix, diagonal=1)
    stitches = []
    
    for i in range(num_stitches):
        index = (simi_matrix == torch.max(simi_matrix)).nonzero()
        stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
        simi_matrix[index[0, 0], :] = -float("inf")
        simi_matrix[index[0, 1], :] = -float("inf")
        simi_matrix[:, index[0, 0]] = -float("inf")
        simi_matrix[:, index[0, 1]] = -float("inf")
    
    if len(stitches) == 0:
        stitches = None
    else:
        stitches = np.array(stitches)
        if stitches.shape[0] != 2:
            stitches = np.transpose(stitches, (1, 0))
    return stitches


def spec_to_boxmesh(spec_path):
    pattern = NNSewingPattern(spec_path, view_ids=False)