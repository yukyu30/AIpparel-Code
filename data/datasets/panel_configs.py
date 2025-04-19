from dataclasses import dataclass, field
from typing import List, Optional
from torch import Tensor
import torch

@dataclass
class PanelConfig():
    max_pattern_len: Optional[int] 
    max_panel_len : Optional[int]
    max_num_stitches: Optional[int]
    max_stitch_edges: Optional[int]
    panel_classification: Optional[str]
    filter_by_params : Optional[str]

@dataclass 
class PanelStats():
    outlines: List[float]
    rotations: List[float]
    stitch_tags: List[float]
    translations: List[float]
    
@dataclass
class StatsConfig():
    scale: List[float]
    shift: List[float]

@dataclass 
class StandardizeConfig():
    outlines: StatsConfig = field(default_factory=StatsConfig)
    rotations: StatsConfig = field(default_factory=StatsConfig)
    stitch_tags: StatsConfig = field(default_factory=StatsConfig)
    translations: StatsConfig = field(default_factory=StatsConfig)
    vertices: StatsConfig = field(default_factory=StatsConfig)