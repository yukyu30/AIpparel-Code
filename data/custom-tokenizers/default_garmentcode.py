
import torch, numpy as np, os, sys
from pathlib import Path 


# TODO What is this part doing? 
os.system('export PYTHONPATH=""')
currentdir = os.path.dirname(os.path.realpath(__file__))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(0, grandparentdir)
root_path = os.path.dirname(os.path.dirname(os.path.abspath(parentdir)))


from data.datasets.garmentcodedata.garmentcode_dataset import GarmentCodeData
from data.datasets.garmentcodedata.pattern_converter import NNSewingPattern, InvalidPatternDefError
from IPython.display import Image
import tiktoken
from data.datasets.utils import PanelEdgeType, SpecialTokens
from data.datasets.panel_configs import *




class DefaultGarmentCode: 
    def __init__(self, 
                 standardize: StandardizeConfig,
                 text_encoder = 'gpt2',
                 bin_size = 256,
                 base_offset = 0,
                 ):
        self.text_encoder = tiktoken.get_encoding(text_encoder)
        self.bin_size = bin_size
        self.special_token_offset = base_offset
        self.panel_edge_type_offset = self.special_token_offset + len(SpecialTokens)
        self.edge_offset = len(PanelEdgeType) + self.panel_edge_type_offset
        self.total_vocab_size = self.edge_offset + bin_size
        self.gt_stats = standardize



    def encode(self, item):
        name = item['name']
        features = item['features']
        ground_truth = item['ground_truth']
        data_folder = item['data_folder']

        outlines = ground_truth['outlines'].detach().numpy()
        num_edges = ground_truth['num_edges'].detach().numpy()
        num_panels = ground_truth['num_panels'].detach().numpy()

        panel_list = self._panel_as_list(outlines, num_edges, num_panels)
        

        out_tokens = [SpecialTokens.PATTERN_START.value + self.special_token_offset]
        for panel_edges in panel_list:
            panel_tokens = [SpecialTokens.PANEL_START.value + self.special_token_offset]
            for panel_edge in panel_edges:
                edge_type: PanelEdgeType = panel_edge[0]
                param_num = edge_type.get_num_params()
                edge_params: np.ndarray = (panel_edge[1]  - self.gt_stats.outlines.shift[:param_num]) / self.gt_stats.outlines.scale[:param_num]
                edge_params = np.clip(edge_params, 0, 1) * self.bin_size
                edge_params = edge_params.astype(int).clip(0, self.bin_size - 1) + self.edge_offset
                panel_tokens.append(edge_type.value + self.panel_edge_type_offset)
                if param_num > 0:
                    panel_tokens.extend(edge_params.flatten()[:param_num].tolist())
            panel_tokens.append(SpecialTokens.PANEL_END.value + self.special_token_offset)
            out_tokens.extend(panel_tokens)
        out_tokens.append(SpecialTokens.PATTERN_END.value + self.special_token_offset)
        return out_tokens
    

    def _panel_as_list(self, outlines, num_edges, num_panels):
        panel_indices = np.where(num_edges > 0)[0]
        num_edges_prime = num_edges[panel_indices]
        outlines=outlines[panel_indices]
        panel_list = []
        for i in range(num_panels):
            panel = outlines[i][:num_edges_prime[i]]
            all_edges = []

            # currently appending all edges, including the last one 
            # can change to appending the last edge seperately 
            for j in range(num_edges_prime[i]):
                edge = panel[j]
                if edge[-1]: 
                    edge_type = PanelEdgeType.ARC 
                    params = edge[:-2]
                elif np.all(edge[2:6] == 0):
                    edge_type = PanelEdgeType.LINE
                    params = edge[:2]
                elif np.all(edge[4:6] == 0): 
                    edge_type = PanelEdgeType.CURVE
                    params = edge[:4]
                else: 
                    edge_type = PanelEdgeType.CUBIC_CURVE
                    params = edge[:6]
                all_edges.append([edge_type, params])
            panel_list.append(all_edges)

        return panel_list
    
    

    def decode(self, token_sequence: np.array): 
        # find the panel starts and ends
        # intial sanity checks on the panel (more starts than finishes, more finishes than starts, etc.)
        garment_end = np.where(token_sequence == SpecialTokens.PATTERN_END.value + self.special_token_offset)[0]
        if len(garment_end) > 0:
            token_sequence = token_sequence[:garment_end[0]]
        else:
            token_sequence = token_sequence
        garment_start = np.where(token_sequence == SpecialTokens.PATTERN_START.value + self.special_token_offset)[0]    
        if len(garment_start) > 0:
            try:
                pattern_description = self.text_encoder.decode(token_sequence[:garment_start[0]].tolist())
            except:
                pattern_description = None
            token_sequence = token_sequence[garment_start[0]+1:]
        else:
            pattern_description = None
        panel_starts = np.where(token_sequence == SpecialTokens.PANEL_START.value + self.special_token_offset)[0]
        panel_ends = np.where(token_sequence == SpecialTokens.PANEL_END.value + self.special_token_offset)[0]

        n_panels = min(len(panel_starts), len(panel_ends))
        panels = []
        panel_names = ["NONE" for _ in range(n_panels)]
        if n_panels == 0:
            return panels, panel_names, pattern_description

        if len(panel_starts) < len(panel_ends):
            panel_ends = panel_ends[:len(panel_starts)]

        if len(panel_starts) > len(panel_ends):
            panel_starts = panel_starts[:len(panel_ends)]

        if np.any(panel_starts > panel_ends):
            return panels, panel_names, pattern_description
        


        current_mark = 0
        for i in range(n_panels):
            panel_start = panel_starts[i]
            panel_end = panel_ends[i]
            try: 
                panel_description = self.text_encoder.decode(token_sequence[current_mark:panel_start].tolist())
            except:
                panel_description = ''
            panel_names[i] = panel_description if panel_description != '' else "NONE"
            panel = token_sequence[panel_start+1:panel_end]
            commands = np.isin(panel, [e.value + self.panel_edge_type_offset for e in PanelEdgeType]).nonzero()[0]
            last_point = np.array([0, 0])
            all_edges = []
            for j in range(len(commands)):
                command = commands[j]
                command_end = commands[j+1] if j+1 < len(commands) else len(panel)
                edge_type = PanelEdgeType(panel[command].item() - self.panel_edge_type_offset)
                edge_params = panel[command+1:command_end]
                num_params = edge_type.get_num_params()
                padding = np.zeros(7 - num_params)
                if len(edge_params) != num_params:
                    # force close the loop and break out of this panel
                    edge_params = np.concatenate([-last_point, np.array([0, 0])])
                    all_edges.append(edge_params)
                    break

                # closure loops are ignored for the moment 
                # if edge_type == PanelEdgeType.CLOSURE_LINE:
                #     # Start point is always 0
                #     edge_params = np.concatenate([-last_point, np.array([0, 0])])
                    
                else: 
                    edge_params = edge_params.astype(float)
                    edge_params = (edge_params - self.edge_offset + 0.5).clip(0, self.bin_size) / self.bin_size
                    edge_params = (edge_params * np.array(self.gt_stats.outlines.scale[:num_params]) 
                                               + np.array(self.gt_stats.outlines.shift[:num_params]))
                    if edge_type == PanelEdgeType.ARC: 
                        padding[-1] = 1
                    edge_params = np.concatenate([edge_params, padding])

                # closure loops are ignored for the moment 
                # elif edge_type == PanelEdgeType.CLOSURE_CURVE:
                #     edge_params = edge_params.astype(float)
                #     edge_params = (edge_params - self.edge_offset + 0.5).clip(0, self.bin_size) / self.bin_size
                #     edge_params = edge_params * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                #     ctrl_pt = NNSewingPattern._control_to_relative_coord(last_point, np.array([0, 0]), edge_params)
                #     edge_params = np.concatenate([-last_point, ctrl_pt])
                all_edges.append(edge_params)
            all_edges = np.stack(all_edges).astype(float)
            panels.append(all_edges)
            current_mark = panel_end + 1
        max_edge_len = max(len(edges) for edges in panels)
        out_panel_tensor = np.zeros((n_panels, max_edge_len, 7))
        for i, edges in enumerate(panels):
            out_panel_tensor[i, :len(edges)] = edges
        return out_panel_tensor, panel_names, pattern_description
        


if __name__ == "__main__":
    root_dir = "/miele/timur/garmentcodedata"
    dataset = GarmentCodeData(root_dir, start_config={'random_pairs_mode': False, 'body_type' : 'default_body',
                        'panel_classification' : '/home/timur/sewformer-garments/garment-estimator/nn/data_configs/panel_classes.json',
                        'max_pattern_len' : 37,                            
                        'max_panel_len' : 40,
                        'max_num_stitches' : 108,
                        'mesh_samples' : 2000, 
                        'max_datapoints_per_folder' : 10,                          
                        'stitched_edge_pairs_num': 100, 'non_stitched_edge_pairs_num': 100,
                        'shuffle_pairs': False, 'shuffle_pairs_order': False, 
                        'data_folders': [
                        "garments_5000_0",
                        # "garments_5000_1", "garments_5000_2", "garments_5000_3", "garments_5000_4", "garments_5000_5", "garments_5000_6", "garments_5000_7", "garments_5000_8"
                        ]})
    
    conf = StandardizeConfig(
        outlines=StatsConfig(shift=[-51.205456, -49.627438, 0.0, -0.35, 0.0, -0.35, 0.0], scale=[104.316666, 99.264435, 0.5, 0.7, 0.8, 0.7, 1.0]),
        rotations=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        stitch_tags=StatsConfig(shift=[0, 0], scale=[0., 0.5]),
        translations=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        vertices=StatsConfig(shift=[-30, -70], scale=[190, 180])
    )

    garment_code = DefaultGarmentCode(standardize=conf)
    tokens = garment_code.encode(dataset, 0)

    print(tokens)

    tensor, names, description = garment_code.decode(np.array(tokens))

    