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

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    if len(anno) == 0:
        return False
    if _has_only_empty_bbox(anno):
        return False
    return True

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class VisualGenome:

    def __init__(self, root, ann_file=None, filter_classes=True, cats_file=None):
        
        self.anns = dict()
        self.imgs = dict()
        self.cats = dict()
        self.ids = list()
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)
        self.root = root

        if not ann_file == None:
            print('Loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(ann_file, 'r'))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
        if not cats_file == None:
            categories = json.load(open(cats_file, 'r'))
        if not ann_file == None and not cats_file == None:
            if filter_classes:
                dataset_coco_classes = self.get_only_coco_classes(dataset, categories)
                dataset = dataset_coco_classes
            self.createIndex(dataset, categories)
            self.ids = list(sorted(self.imgs.keys()))
            #self.add_img_info()

    def __getitem__(self, index):

        img_id = self.ids[index]
        ann_ids = self.getAnnIds(imgIds=img_id)
        target = self.loadAnns(ann_ids)
        path = self.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        return img, target

    def __len__(self):
        
        return len(self.ids)

    def get_only_coco_classes(self, dataset, categories):

        cats = list()
        for cat_id, cat_name in categories.items():
            cats.append(cat_name)

        dataset_coco_classes = list()
        for image in dataset:
            obj_list = []
            for object_dic in image['objects']:
                try:
                    obj_class = str(object_dic['names'][0])
                except IndexError:
                    continue
                if obj_class in cats:
                    obj_list.append(object_dic) 
            if len(obj_list)>0:
                img_dic = {}
                img_dic['image_id'] = image['image_id']
                img_dic['objects'] = obj_list
                try:
                    img_dic['image_url'] = image['image_url']
                except:
                    continue
                dataset_coco_classes.append(img_dic)

        return dataset_coco_classes

    def createIndex(self, dataset, categories):

        print('Creating index...')
        anns = {}
        imgs = {}
        cats = {}
        imgToAnns = defaultdict(list)
        catToImgs = defaultdict(list)
        
        for cat_id in categories.keys():
            cat = {'id': cat_id,
                   'name': categories[cat_id],
                   'supercategory': ''}
            cats[cat['id']] = cat
        
        for image in dataset:
            try:
                w, h = VisualGenome.__getitem__(self, image['image_id'])[0].size
            except:
                continue
            for ann in image['objects']:
                for cat_id, cat_name in categories.items():
                    if cat_name == ann['names'][0]:
                        cat = cat_id

                ann = {'id': ann['object_id'],
                        'image_id': image['image_id'],
                        'category_id': cat,
                        'bbox': [ann['x'], ann['y'], ann['w'], ann['h']]} 
                anns[ann['id']] = ann
                imgToAnns[ann['image_id']].append(ann)
                catToImgs[ann['category_id']].append(ann['image_id'])
            #width, height = self.__getitem__(image['image_id'])[0].size
            img = {'id': image['image_id'], 
                    'width': w,
                    'height': h,
                    'file_name': str(image['image_id']) + '.jpg'}
            imgs[image['image_id']] = img
        print('Index created!')

        # Assign to class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def add_img_info(self):

        for img_id, img in self.imgs.items():
            print(img_id)
            w, h = VisualGenome.__getitem__(self, img_id)[0].size
            img['width'] = w
            img['height'] = h
    
    def getAnnIds(self, imgIds=[], catIds=[]):
        
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            anns = self.anns
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.anns
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
        ids = [ann['id'] for ann in anns]
        return ids
    
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.cats.values()
        else:
            cats = self.cats
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)
        
    def loadAnns(self, ids=[]):

        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]
    
    def loadCats(self, ids=[]):
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns, draw_bbox=False):
        if len(anns)==0:
            return 0
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        for ann in anns:
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            if draw_bbox:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4,2))
                polygons.append(Polygon(np_poly))
                color.append(c)
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

    def loadRes(self, resFile):
        res = VisualGenome()
        res.imgs = [img for img in self.imgs]

        print('Loading and preparing results')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'Results are not an array of aobjects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
                               'Results do not correspond to current Visual Genome set'
        if 'bbox' in anns[0] and not anns[0]['bbox']==[]:
            res.cats = copy.deepcopy(self.cats)
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                ann['id'] = id+1
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))
        res.anns = anns
        res.createIndex(dataset, categories) #####
        return res

    def loadNumpAnnotations(self, data):

        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{'image_id'  : int(data[i, 0]),
                     'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                     'score' : data[i, 5],
                     'category_id': int(data[i, 6])}]
        return ann


