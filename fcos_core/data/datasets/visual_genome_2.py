# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from fcos_core.structures.bounding_box import BoxList

from .vg_detection import VisualGenomeDetection

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    if len(anno) == 0:
        return False
    if _has_only_empty_bbox(anno):
        return False
    return True

class VisualGenomeDataset(VisualGenomeDetection):

    def __init__(self, ann_file, cats_file, root, remove_images_without_annotations, vg_format, filter_classes, transforms=None):
    
        super(VisualGenomeDataset, self).__init__(root, ann_file, cats_file, vg_format, filter_classes)
        self.ids = sorted(self.ids)

        # Filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.vg.getAnnIds(self, imgIds=img_id)
                anno = self.vg.loadAnns(self, ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(self.vg.getCatIds())}
        self.contiguous_category_id_to_json_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):

        #img, anno = VisualGenomeDetection.__getitem__(self, idx)
        img, anno = super(VisualGenomeDataset, self).__getitem__(idx)

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1,4)
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [obj['category_id'] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        target = target.clip_to_image(remove_empty=True)
        if self._transforms is not None:
            img, target = self._transforms(img, target) 

        return img, target, idx

    def get_img_info(self, index):
        
        img_id = self.id_to_img_map[index]
        img_data = self.vg.imgs[img_id]
        
        return img_data

