import json
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from .data_aug import RandomAugmentation


def build_transform(dataset_type, cfg):
    lower_type = dataset_type.lower()
    interpolation_method = InterpolationMode[cfg.INTERPOLATION.upper()]
    transform = None
    if cfg.NO_TRANSFORM:
        return transform
    if "train" in lower_type:
        transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(img_size)
                transforms.Resize(cfg.SIZE, interpolation=interpolation_method),
                RandomAugmentation(cfg, interpolation_method),
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(cfg.PIXEL_MEAN),
                    tuple(cfg.PIXEL_STD),
                ),
            ]
        )
    elif "val" in lower_type or "test" in lower_type:
        transform = transforms.Compose(
            [
                transforms.Resize(cfg.SIZE, interpolation=interpolation_method),
                transforms.ToTensor(),
                transforms.Normalize(
                    tuple(cfg.PIXEL_MEAN),
                    tuple(cfg.PIXEL_STD),
                ),
            ]
        )
    return transform

class SpineDataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "Acute Vertebral Compression Fracture",
            "Chronic Vertebral Compression Fracture",
            "Burst Fracture",
            "Pars Interarticularis Defect",
            "Transverse Process Fracture",
            "Posterior Vertebral Wall High Signal",
            "Lumbar Disc Herniation",
            "Lumbar Spinal Canal Stenosis",
            "Lateral Recess Stenosis",
            "Foraminal Stenosis",
            "Lumbar Spondylolisthesis",
            "Lumbar Scoliosis",
            "Adjacent Segment Disease",
            "Schmorl's Node",
            "Kummell Disease",
            "Modic Endplate Changes Modic",
            "Spondylitis",
            "Spinal Infection",
            "Subcutaneous Fasciitis",
            "Multiple Myeloma",
            "Vertebral Body Tumor",
            "Intradural Spinal Tumor",
            "Postoperative Internal Fixation",
            "Non-instrumented Postoperative Spine",
            "Postoperative Percutaneous Kyphoplasty",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)  # Shuffle to randomize data
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)
        self.tran = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        h5_path = os.path.join(self.root, item["study_h5_path"])
        ap_path = os.path.join(self.root, item["ap_jpg_path"])
        lap_path = os.path.join(self.root, item["lap_jpg_path"])
        if os.path.commonpath([ap_path]) == os.path.commonpath([self.root]):
            ap_path = lap_path
        if os.path.commonpath([lap_path]) == os.path.commonpath([self.root]):
            lap_path = ap_path
        # if item["ap_jpg_path"] == 'NULL':
        #     ap_path = lap_path
        # if item["lap_jpg_path"] == 'NULL':
        #     lap_path = ap_path

        ap_img = Image.open(ap_path).convert("RGB")
        lap_img = Image.open(lap_path).convert("RGB")

        if self.transform is not None:
            ap_img = self.transform(ap_img)
            lap_img = self.transform(lap_img)

        text = item["text"]
        labels = torch.Tensor(item["labels"])
        target = labels

        return h5_path, ap_img, lap_img, text, target

    def name(self):
        return "Spine"
class VindrSpineDataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "Disc space narrowing",
            "Foraminal stenosis",
            "Osteophytes",
            "Spondylolysthesis",
            "Surgical implant",
            "Vertebral collapse",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)  # Shuffle to randomize data
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)
        self.tran = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        h5_path = os.path.join(self.root, item["study_h5_path"])
        ap_path = os.path.join(self.root, item["ap_jpg_path"])
        lap_path = os.path.join(self.root, item["lap_jpg_path"])
        if os.path.commonpath([ap_path]) == os.path.commonpath([self.root]):
            ap_path = lap_path
        if os.path.commonpath([lap_path]) == os.path.commonpath([self.root]):
            lap_path = ap_path
        # if item["ap_jpg_path"] == 'NULL':
        #     ap_path = lap_path
        # if item["lap_jpg_path"] == 'NULL':
        #     lap_path = ap_path

        ap_img = Image.open(ap_path).convert("RGB")
        lap_img = Image.open(lap_path).convert("RGB")

        if self.transform is not None:
            ap_img = self.transform(ap_img)
            lap_img = self.transform(lap_img)

        text = item["text"]
        labels = torch.Tensor(item["labels"])
        target = labels

        return h5_path, ap_img, lap_img, text, target

    def name(self):
        return "VindrSpine"




class BuulSpineDataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "Right Laterolisthesis",
            "Anterolisthesis",
            "Retrolisthesis",
            "Left Laterolisthesis",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)  # Shuffle to randomize data
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)
        self.tran = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        h5_path = os.path.join(self.root, item["study_h5_path"])
        ap_path = os.path.join(self.root, item["ap_jpg_path"])
        lap_path = os.path.join(self.root, item["lap_jpg_path"])
        if os.path.commonpath([ap_path]) == os.path.commonpath([self.root]):
            ap_path = lap_path
        if os.path.commonpath([lap_path]) == os.path.commonpath([self.root]):
            lap_path = ap_path
        # if item["ap_jpg_path"] == 'NULL':
        #     ap_path = lap_path
        # if item["lap_jpg_path"] == 'NULL':
        #     lap_path = ap_path

        ap_img = Image.open(ap_path).convert("RGB")
        lap_img = Image.open(lap_path).convert("RGB")

        if self.transform is not None:
            ap_img = self.transform(ap_img)
            lap_img = self.transform(lap_img)

        text = item["text"]
        labels = torch.Tensor(item["labels"])
        target = labels

        return h5_path, ap_img, lap_img, text, target

    def name(self):
        return "BuulSpine"


class SPIDERDataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "Modic",
            "UP endplate",
            "LOW endplate",
            "Spondylolisthesis",
            "Disc herniation",
            "Disc narrowing",
            "Disc bulging",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)  # Shuffle to randomize data
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)
        self.tran = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        h5_path = os.path.join(self.root, item["study_h5_path"])
        ap_path = os.path.join(self.root, item["ap_jpg_path"])
        lap_path = os.path.join(self.root, item["lap_jpg_path"])
        if os.path.commonpath([ap_path]) == os.path.commonpath([self.root]):
            ap_path = lap_path
        if os.path.commonpath([lap_path]) == os.path.commonpath([self.root]):
            lap_path = ap_path
        # if item["ap_jpg_path"] == 'NULL':
        #     ap_path = lap_path
        # if item["lap_jpg_path"] == 'NULL':
        #     lap_path = ap_path

        ap_img = Image.open(ap_path).convert("RGB")
        lap_img = Image.open(lap_path).convert("RGB")

        if self.transform is not None:
            ap_img = self.transform(ap_img)
            lap_img = self.transform(lap_img)

        text = item["text"]
        labels = torch.Tensor(item["labels"])
        target = labels

        return h5_path, ap_img, lap_img, text, target

    def name(self):
        return "SPIDER"


class RSNADataset(data.Dataset):

    def __init__(
        self,
        root,
        dataset_split,
        dataset_map_file="",
        proportion=1,
        transform_cfg=None,
    ):
        self.root = root
        self.classnames = [
            "Spinal Canal Stenosis",
            "Right Neural Foraminal Narrowing",
            "Left Neural Foraminal Narrowing",
            "Right Subarticular Stenosis",
            "Left Subarticular Stenosis",
        ]

        with open(dataset_map_file, "r", encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file]

        if dataset_split == "train":
            np.random.shuffle(self.data)  # Shuffle to randomize data
            num_examples = len(self.data)
            pick_example = int(num_examples * proportion)
            self.data = self.data[:pick_example]
        else:
            self.data = self.data
        self.transform = build_transform(dataset_split, transform_cfg)
        self.tran = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        h5_path = os.path.join(self.root, item["study_h5_path"])
        ap_path = os.path.join(self.root, item["ap_jpg_path"])
        lap_path = os.path.join(self.root, item["lap_jpg_path"])
        if os.path.commonpath([ap_path]) == os.path.commonpath([self.root]):
            ap_path = lap_path
        if os.path.commonpath([lap_path]) == os.path.commonpath([self.root]):
            lap_path = ap_path
        # if item["ap_jpg_path"] == 'NULL':
        #     ap_path = lap_path
        # if item["lap_jpg_path"] == 'NULL':
        #     lap_path = ap_path

        ap_img = Image.open(ap_path).convert("RGB")
        lap_img = Image.open(lap_path).convert("RGB")

        if self.transform is not None:
            ap_img = self.transform(ap_img)
            lap_img = self.transform(lap_img)

        text = item["text"]
        labels = torch.Tensor(item["labels"])
        target = labels

        return h5_path, ap_img, lap_img, text, target

    def name(self):
        return "RSNA"