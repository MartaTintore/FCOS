# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import itertools
import copy
import time
import json
import os
import numpy as np

from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from fcos_core.structures.bounding_box import BoxList
from .vg import VisualGenome

class VisualGenomeDataset(VisualGenome):

    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None, cats_file=None, filter_classes=True):
    
        super(VisualGenomeDataset, self).__init__(root, ann_file, filter_classes, cats_file)
        self.ids = sorted(self.ids)

        # Filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = VisualGenome.getAnnIds(self, imgIds=img_id)
                anno = VisualGenome.loadAnns(self, ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids


        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(VisualGenome.getCatIds(self))}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):

        img, anno = VisualGenome.__getitem__(self, idx)

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1,4)
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target) 

        return img, target, idx

    def get_img_info(self, index):
        
        img_id = self.id_to_img_map[index]
        img_data = self.imgs[img_id]
        
        return img_data

