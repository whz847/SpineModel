import os
from .cxr_dataset import (
    SpineDataset,
    VindrSpineDataset,
    BuulSpineDataset,
    SPIDERDataset,
    RSNADataset,
)

MODEL_TABLE = {
    "first_XMRI": SpineDataset,
    "VindrSpine": VindrSpineDataset,
    "BuulSpine": BuulSpineDataset,
    "SPIDER": SPIDERDataset,
    "RSNA": RSNADataset,
}


def build_dataset(cfg, dataset_name, dataset_split, dataset_map_file=""):
    print("Building Dataset...")
    print("dataset_root = %s" % cfg.DATASET.ROOT)
    print("dataset_split = %s" % dataset_split)
    print("dataset proportion = %f" % cfg.DATALOADER.TRAIN.PROPORTION)
    if dataset_map_file != "":
        dataset_map_file = os.path.join(cfg.DATASET.ROOT, dataset_map_file)
    image_size = cfg.INPUT.SIZE[0]
    print("image_input_size = %d" % image_size)
    return MODEL_TABLE[dataset_name](
        cfg.DATASET.ROOT,
        dataset_split,
        dataset_map_file=dataset_map_file,
        proportion=cfg.DATALOADER.TRAIN.PROPORTION,
        transform_cfg=cfg.INPUT,
    )
