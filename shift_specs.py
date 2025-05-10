import os
from tqdm import tqdm
import glob
import json
import numpy as np 
from data.patterns.pattern_converter import NNSewingPattern
from data.patterns.panel_classes import PanelClasses
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor

gcd_path = "[PATH TO GCD]"
panel_classifier = PanelClasses(classes_file="assets/data_configs/panel_classes_garmentcodedata.json")

def process_datapoint(pattern_folder):
    data_name = pattern_folder.split('/')[-1]
    spec_file = os.path.join(pattern_folder, f'{data_name}_specification.json')
    dump_spec_name = f"{data_name}_specification_shifted.json"
    dump_file = os.path.join(pattern_folder, dump_spec_name)
    if os.path.exists(dump_file):
        return
    if not os.path.exists(spec_file):
        return
    gt_pattern = NNSewingPattern(spec_file, panel_classifier=panel_classifier, template_name=data_name)
    gt_pattern.name = data_name
    for key in gt_pattern.pattern['panels'].keys():
        panel = gt_pattern.pattern['panels'][key]
        vertices = panel['vertices']
        offset = vertices[0]
        if offset != [0, 0]:
            offset = np.array(offset)
            new_vertices = np.array(vertices) - offset
            panel['vertices'] = new_vertices.tolist()
            translation = panel['translation']
            shift = np.append(offset, 0)  # to 3D
            panel_rotation = Rotation.from_euler('xyz', panel['rotation'], degrees=True)
            shift = panel_rotation.as_matrix().dot(shift)
            translation = [translation[i] + shift[i] for i in range(3)]
            panel['translation'] = translation
    
    with open(dump_file, 'w') as f:
        json.dump(gt_pattern.pattern, f, indent=2)

all_patterns = glob.glob(os.path.join(gcd_path, '*', "default_body", "*"))[:2]
with ThreadPoolExecutor(max_workers=32) as executor:
    list(tqdm(executor.map(process_datapoint, all_patterns), total=len(all_patterns)))
