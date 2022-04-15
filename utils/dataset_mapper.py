# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import detectron2.structures.masks as MASKS
from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import torchvision.transforms as TSF
from PIL import Image, ImageFilter
"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


import random


class temporary_seed:
    def __init__(self, seed):
        self.seed = seed
        self.backup = None

    def __enter__(self):
        self.backup = np.random.randint(2**32-1, dtype=np.uint32)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, *_):
        np.random.seed(self.backup)
        torch.manual_seed(self.backup)
        random.seed(self.backup)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@torch.no_grad()
def mask_index_encode(boxlist, output_h, output_w):
    # transform polymask to seg map style inputs
    # this is for detectron 2, mask_color_encode should be very similar to this version
    _DEVICE = boxlist.gt_boxes.device

    labels = boxlist.gt_classes  # 0 for background
    H, W = output_h, output_w
    output = torch.zeros(H, W, dtype=torch.long)

    if not hasattr(boxlist, 'gt_masks'):
        return output.to(_DEVICE)

    masks = boxlist.gt_masks
    for label, mask in zip(labels, masks):
        # there could be overlap, but we just ignore them
        mask = MASKS.polygons_to_bitmask(mask, H, W)
        output[mask] = label  # fill in index

    # pad as the same logic as image, remember to move it to GPU
    return output.to(_DEVICE)


def box_color_encode(boxlist, output_h, output_w, num_classes, target_noise=True, bg_nosie=False):
    # TODO change to matrix operations! TOO ugly now
    # print(boxlist._fields.keys())
    output = torch.randn(num_classes, output_h, output_w,
                         dtype=torch.float, device=boxlist.gt_boxes.device)
    if not bg_nosie:
        output.fill_(0)
    else:
        output.mul_(0.05)
    bboxs = boxlist.gt_boxes
    labels = boxlist.gt_classes - 1
    for bbox, label in zip(bboxs, labels):
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        y = torch.arange(0, output_h, dtype=torch.float, device=output.device)
        x = torch.arange(0, output_w, dtype=torch.float, device=output.device)
        y, x = torch.meshgrid(y, x)
        color = 1 - torch.max(torch.abs(x - cx) / w, torch.abs(y - cy) / h)
        if not target_noise:
            color = (color >= 0.5).float()
        else:
            color = color * (color >= 0.5).float()
            color = color * (torch.rand(1, dtype=color.dtype,
                                        device=color.device) * 2).clamp(max=1)
        output[label] = torch.max(output[label], color)

    return output


def box_mask(boxlist, output_h, output_w):
    # TODO change to matrix operations! TOO ugly now
    # print(boxlist._fields.keys())
    output = torch.zeros(output_h, output_w,
                         dtype=torch.bool, device=boxlist.gt_boxes.device)

    bboxs = boxlist.gt_boxes
    labels = boxlist.gt_classes - 1
    for bbox, label in zip(bboxs, labels):
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        y = torch.arange(0, output_h, dtype=torch.float, device=output.device)
        x = torch.arange(0, output_w, dtype=torch.float, device=output.device)
        y, x = torch.meshgrid(y, x)
        color = 1 - torch.max(torch.abs(x - cx) / w, torch.abs(y - cy) / h)
        color = color >= 0.5
        output[color] = True
    return output


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        load_map: bool = False,
        map_dim: int = 32,
        cfg=None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        self.cfg = cfg
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.map_dim = map_dim
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        # incremental
        self.load_map = load_map
        self.load_label = cfg.MODEL.LOAD_BOXMAP
        self.load_obj_box_mask = cfg.MODEL.LOAD_BOX_MASK
        self.stronger_augments = cfg.MODEL.STRONGER_AUGS
        if self.stronger_augments:
            self.extra_augs = TSF.Compose([
                    TSF.RandomApply([
                        TSF.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    TSF.RandomGrayscale(p=0.2),
                    TSF.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                ])

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(
                cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "cfg": cfg
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        if 'LOAD_LABELMAP' in cfg.MODEL and cfg.MODEL.LOAD_LABELMAP:
            #print('LOADING SEG MAPS!')
            ret['load_map'] = True
            ret['map_dim'] = cfg.MODEL.DISTILLATOR.HIDDEN_DIM
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(
            dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop(
                "sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(
                sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask and not self.load_map:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            if self.load_map:
                #label_map = box_color_encode_rand_affine(dataset_dict["instances"], image_shape[0], image_shape[1], self.cfg.MODEL.ROI_HEADS.NUM_CLASSES, self.map_dim)
                label_map = mask_index_encode(
                    dataset_dict["instances"], image_shape[0], image_shape[1])
                dataset_dict['label_map'] = label_map

            if self.load_label:
                label_box = box_color_encode(dataset_dict["instances"], image_shape[0], image_shape[1],
                                             self.cfg.MODEL.ROI_HEADS.NUM_CLASSES, target_noise=self.cfg.MODEL.DISTILLATOR.LABEL_TARGET_NOISE)
                dataset_dict['box_map'] = label_box

            if self.stronger_augments:
                extra_images = np.array(self.extra_augs(Image.fromarray(image)))
                dataset_dict["extra_images"] = torch.as_tensor(
                    np.ascontiguousarray(extra_images.transpose(2, 0, 1)))

            if self.load_obj_box_mask:
                label_mask = box_mask(dataset_dict["instances"], image_shape[0], image_shape[1])
                dataset_dict['box_mask'] = label_mask

        return dataset_dict
