# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import itertools
import copy
import time
import json
import os
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from fcos_core.structures.bounding_box import BoxList

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class VisualGenome:

    def __init__(self, root=None, ann_file_vg=None, cats_file=None, vg_format=False, filter_classes=True):
        
        self.dataset = dict()
        self.anns = dict()
        self.imgs = dict()
        self.cats = dict()
        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)

        if vg_format and (not ann_file_vg == None and not cats_file == None):
            categories_vg = json.load(open(cats_file, 'r'))
            print('Loading vg annotations into memory...')
            tic = time.time()
            dataset_vg = json.load(open(ann_file_vg, 'r'))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            if filter_classes and not root == None:
                print('Filtering visual genome dataset (only coco classes)...')                
                print('Initial dataset contains {} images'.format(len(dataset_vg)))
                dataset_vg = self.get_only_coco_classes(root, dataset_vg, categories_vg)
                print('Filtered dataset contains {} images'.format(len(dataset_vg)))
            self.to_coco_format(dataset_vg, categories_vg)
        if not ann_file_vg == None:
            self.createIndex()
            if vg_format and not root == None:
            	self.add_img_info(root)

    def to_coco_format(self, dataset_vg, categories_vg):

        self.dataset['info'] = {}
        self.dataset['licenses'] = []
        self.dataset['images'] = []
        self.dataset['annotations'] = []
        self.dataset['categories'] = []

        for cat_id in categories_vg.keys():
            cat = {'id': cat_id,
                   'name': categories_vg[cat_id],
                   'supercategory': ''}
            self.dataset['categories'].append(cat)

        for image in dataset_vg:
            img = {}
            img['id'] = image['image_id']
            img['file_name'] = str(image['image_id']) + '.jpg'
            self.dataset['images'].append(img)
            for box in image['objects']:
                ann = {}
                ann['id'] = box['object_id']
                ann['image_id'] = image['image_id']
                cat_dic = list(filter(lambda cat: cat['name'] == box['names'][0], self.dataset['categories']))
                ann['category_id'] = cat_dic[0]['id']
                ann['area'] = box['w']*box['h']
                ann['bbox'] = [box['x'], box['y'], box['w'], box['h']]
                self.dataset['annotations'].append(ann)

    def get_only_coco_classes(self, root, dataset_vg, categories_vg):

        cats = list()
        for cat_id, cat_name in categories_vg.items():
            cats.append(cat_name)

        dataset_coco_classes = list()
        for image in dataset_vg:
            #Check image in visual genome folder
            filename = os.path.join(root, str(image['image_id']) + '.jpg')
            if not os.path.isfile(filename):
                continue 
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

    def add_img_info(self, root):
        
        for idx in range(len(self.imgs.keys())):
            img_id = list(self.imgs.keys())[idx]
            path = self.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(root, path)).convert('RGB')
            w, h = image.size
            self.imgs[img_id]['width'] = w
            self.imgs[img_id]['height'] = h

    def createIndex(self):
        
        print('Creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list), defaultdict(list)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
    
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        ids = [ann['id'] for ann in anns]
        return ids
    
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
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

    def loadRes(self, resFile):

        res = VisualGenome()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'Results are not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
                               'Results do not correspond to current Visual Genome set'
        if 'bbox' in anns[0] and not anns[0]['bbox']==[]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                ann['id'] = id+1
                ann['area'] = bb[2]*bb[3]
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
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


