from .garment_detr_2d import build as build_former
from .pcd2garment.garment_pcd import build as build_former_pcd
from .garment_backbone import build as build_backbone
from .diffusion.ldm import build as build_ldm

def build_model(args):
    if args["NN"]["model"] == "GarmentBackbone":
        return build_backbone(args)
    elif args["NN"]["model"] == "GarmentPCD":
        return build_former_pcd(args)
    elif args["NN"]["model"] == "LDM":
        return build_ldm(args)
    else:
        return build_former(args)