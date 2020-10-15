# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
import json
from torchvision import transforms as T

from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.structures.image_list import to_image_list
from fcos_core import layers as L
from fcos_core.utils import cv2_util


class VisualGenomeDemo(object):

    # COCO categories for pretty print
    CATEGORIES = [
        "__background","person","bicycle","car","motorcycle","airplane",
        "bus","train","truck","boat","traffic light","fire hydrant",
        "stop sign","parking meter","bench","bird","cat","dog","horse",
        "sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard",
        "surfboard","tennis racket","bottle","wine glass","cup","fork",
        "knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop",
        "mouse","remote","keyboard","cell phone","microwave","oven","toaster",
        "sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"]

    def __init__(self, cfg, confidence_thresholds_for_classes, min_image_size=224, ground_truth_file=None):

        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        self.ground_truth_file = ground_truth_file

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.

        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image, img_name):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        if self.ground_truth_file is not None:
            img_id = int(img_name.split('.')[0])
            gt = self.get_gt(img_id)

        result = image.copy()
        
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)
        if self.ground_truth_file is not None:
            result = self.overlay_boxes_cats_gt(result, gt)
        
        return result

    def get_gt(self, img_id):

        with open(self.ground_truth_file, 'r') as gt_file:
            bboxes_gt = json.load(gt_file)
   
        gt = {'coords': [], 'cats': []}
        for image in bboxes_gt:
            id_img = int(image['image_id'])
            if id_img == img_id:
                for bbox in image['objects']:
                    box_coord = [bbox['x'], bbox['y'], bbox['w'], bbox['h']]
                    cat = bbox['names'][0]
                    gt['coords'].append(box_coord)
                    gt['cats'].append(cat)
        
        return gt

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )

        return image

    def overlay_boxes_cats_gt(self, image, gt):

        """
        Adds the gt boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            gt (dict): ground truth coordinates of bounding boxes
        """
        color = (230,230,250)

        gt_coords = gt['coords']
        gt_cats = gt['cats']

        for box, cat in zip(gt_coords, gt_cats):
            top_left = [box[0], box[1]]
            bottom_right = [box[0]+box[2], box[1]+box[3]]
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 2)
            x, y = box[:2]
            cv2.putText(image, str(cat), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

