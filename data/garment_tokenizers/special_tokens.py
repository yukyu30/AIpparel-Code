from enum import Enum
from typing import Dict, List
class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class DecodeErrorTypes(Enum):
    NO_ERROR = "NO_ERROR"
    UNMATCHED_PATTERN_TOKENS = "UNMATCHED_PATTERN_TOKENS"
    UNMATCHED_PANEL_TOKENS = "UNMATCHED_PANEL_TOKENS"



class SpecialTokens(ExtendedEnum):
    PANEL_START = "<panel_start>"
    PANEL_END = "<panel_end>"
    PATTERN_START = "<pattern_start>"
    PATTERN_END = "<pattern_end>"

class PanelEdgeType(ExtendedEnum):
    MOVE = '<pattern_cmd_MOVE>'
    LINE = '<pattern_cmd_LINE>' 
    CLOSURE_LINE = '<pattern_cmd_CLINE>'
    CURVE = '<pattern_cmd_CURVE>'
    CLOSURE_CURVE = '<pattern_cmd_CCURVE>'
    CUBIC = '<pattern_cmd_CUBIC>'
    CLOSURE_CUBIC = '<pattern_cmd_CCUBIC>'
    ARC = '<pattern_cmd_ARC>'
    CLOSURE_ARC = '<pattern_cmd_CARC>'
    
    def is_closure(self):
        return self in [PanelEdgeType.CLOSURE_LINE, PanelEdgeType.CLOSURE_CURVE, PanelEdgeType.CLOSURE_CUBIC, PanelEdgeType.CLOSURE_ARC]
    
    def is_line(self):
        return self in [PanelEdgeType.LINE, PanelEdgeType.CLOSURE_LINE]
    
    def is_curve(self):
        return self in [PanelEdgeType.CURVE, PanelEdgeType.CLOSURE_CURVE]
    
    def is_cubic_curve(self):  
        return self in [PanelEdgeType.CUBIC, PanelEdgeType.CLOSURE_CUBIC]
    
    def is_arc(self):  
        return self in [PanelEdgeType.ARC, PanelEdgeType.CLOSURE_ARC]
    
    def get_num_params(self):
        if self in [PanelEdgeType.LINE, PanelEdgeType.CLOSURE_CURVE, PanelEdgeType.CLOSURE_ARC]:
            return 2
        elif self in [PanelEdgeType.CURVE, PanelEdgeType.CLOSURE_CUBIC, PanelEdgeType.ARC]:
            return 4
        elif self in [PanelEdgeType.CUBIC]:
            return 6
        elif self == PanelEdgeType.MOVE:
            return 7
        else:
            return 0
    
    def get_closure(self):
        if self == PanelEdgeType.LINE:
            return PanelEdgeType.CLOSURE_LINE
        elif self == PanelEdgeType.CURVE:
            return PanelEdgeType.CLOSURE_CURVE
        elif self == PanelEdgeType.CUBIC:
            return PanelEdgeType.CLOSURE_CUBIC
        elif self == PanelEdgeType.ARC:
            return PanelEdgeType.CLOSURE_ARC
        else:
            return self

class SpecialTokensIndices:
    pattern_start_idx: int
    pattern_end_idx: int
    panel_start_idx: int
    panel_end_idx: int
    
    def __init__(self, token2idx: Dict[str, int]):
        self.pattern_start_idx = token2idx[SpecialTokens.PATTERN_START.value]
        self.pattern_end_idx = token2idx[SpecialTokens.PATTERN_END.value]
        self.panel_start_idx = token2idx[SpecialTokens.PANEL_START.value]
        self.panel_end_idx = token2idx[SpecialTokens.PANEL_END.value]
        
    def get_all_indices(self) -> List[int]:
        return [self.pattern_start_idx, self.pattern_end_idx, self.panel_start_idx, self.panel_end_idx]
        
    def get_token_indices(self, token_type: SpecialTokens) -> int:
        if token_type == SpecialTokens.PATTERN_START:
            return self.pattern_start_idx
        elif token_type == SpecialTokens.PATTERN_END:
            return self.pattern_end_idx
        elif token_type == SpecialTokens.PANEL_START:
            return self.panel_start_idx
        elif token_type == SpecialTokens.PANEL_END:
            return self.panel_end_idx
        
    def get_index_token(self, index: int) -> SpecialTokens:
        if index == self.pattern_start_idx:
            return SpecialTokens.PATTERN_START
        elif index == self.pattern_end_idx:
            return SpecialTokens.PATTERN_END
        elif index == self.panel_start_idx:
            return SpecialTokens.PANEL_START
        elif index == self.panel_end_idx:
            return SpecialTokens.PANEL_END
        
class PanelEdgeTypeIndices:
    move_idx: int
    line_idx: int
    closure_line_idx: int
    curve_idx: int
    closure_curve_idx: int
    cubic_idx: int
    closure_cubic_idx: int
    arc_idx: int
    closure_arc_idx: int
    
    def __init__(self, token2idx: Dict[str, int], rot_as_quat=False):
        self.move_idx = token2idx.get(PanelEdgeType.MOVE.value, -1)
        self.line_idx = token2idx.get(PanelEdgeType.LINE.value, -1)
        self.closure_line_idx = token2idx.get(PanelEdgeType.CLOSURE_LINE.value, -1)
        self.curve_idx = token2idx.get(PanelEdgeType.CURVE.value, -1)
        self.closure_curve_idx = token2idx.get(PanelEdgeType.CLOSURE_CURVE.value, -1)
        self.cubic_idx = token2idx.get(PanelEdgeType.CUBIC.value, -1)
        self.closure_cubic_idx = token2idx.get(PanelEdgeType.CLOSURE_CUBIC.value, -1)
        self.arc_idx = token2idx.get(PanelEdgeType.ARC.value, -1)
        self.closure_arc_idx = token2idx.get(PanelEdgeType.CLOSURE_ARC.value, -1)
        self.rot_as_quat = rot_as_quat
        
    def get_all_indices(self) -> List[int]:
        return [i for i in [
            self.move_idx, 
            self.line_idx, 
            self.closure_line_idx, 
            self.curve_idx, 
            self.closure_curve_idx, 
            self.cubic_idx, 
            self.closure_cubic_idx, 
            self.arc_idx, 
            self.closure_arc_idx
        ] if i != -1]
        
    def get_all_edge_indices(self) -> List[int]:
        return [i for i in [
            self.line_idx, 
            self.closure_line_idx, 
            self.curve_idx, 
            self.closure_curve_idx, 
            self.cubic_idx, 
            self.closure_cubic_idx, 
            self.arc_idx, 
            self.closure_arc_idx
        ] if i != -1]
    
    def get_token_indices(self, token_type: PanelEdgeType) -> int:
        if token_type == PanelEdgeType.MOVE:
            return self.move_idx
        elif token_type == PanelEdgeType.LINE:
            return self.line_idx
        elif token_type == PanelEdgeType.CLOSURE_LINE:
            return self.closure_line_idx
        elif token_type == PanelEdgeType.CURVE:
            return self.curve_idx
        elif token_type == PanelEdgeType.CLOSURE_CURVE:
            return self.closure_curve_idx
        elif token_type == PanelEdgeType.CUBIC:
            return self.cubic_idx
        elif token_type == PanelEdgeType.CLOSURE_CUBIC:
            return self.closure_cubic_idx
        elif token_type == PanelEdgeType.ARC:
            return self.arc_idx
        elif token_type == PanelEdgeType.CLOSURE_ARC:
            return self.closure_arc_idx
        
    def get_index_token(self, index: int) -> PanelEdgeType:
        if index == self.move_idx:
            return PanelEdgeType.MOVE
        elif index == self.line_idx:
            return PanelEdgeType.LINE
        elif index == self.closure_line_idx:
            return PanelEdgeType.CLOSURE_LINE
        elif index == self.curve_idx:
            return PanelEdgeType.CURVE
        elif index == self.closure_curve_idx:
            return PanelEdgeType.CLOSURE_CURVE
        elif index == self.cubic_idx:
            return PanelEdgeType.CUBIC
        elif index == self.closure_cubic_idx:
            return PanelEdgeType.CLOSURE_CUBIC
        elif index == self.arc_idx:
            return PanelEdgeType.ARC
        elif index == self.closure_arc_idx:
            return PanelEdgeType.CLOSURE_ARC
        
    def get_index_param_num(self, index: int) -> int:
        if index == self.move_idx:
            return 6 if not self.rot_as_quat else 7
        return PanelEdgeType(self.get_index_token(index)).get_num_params()