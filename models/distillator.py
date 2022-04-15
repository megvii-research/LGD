# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
# Classes implementation about FCOS and POTO Modified from cvpods (https://github.com/Megvii-BaseDetection/cvpods)
# Copyright (c) Megvii, Inc. All Rights Reserved
# ------------------------------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F
from .customized_detectors import build_customized_detector
from .thirdparty_backbones import build_retinanet_swint_fpn_backbone, build_swint_fpn_backbone
from .adapters import build_adapter
from detectron2.modeling import META_ARCH_REGISTRY
from .base_distillator import BaseDistillator
from detectron2.config import configurable
from typing import List, Tuple, Dict, Optional

import torch.distributed as dist

@META_ARCH_REGISTRY.register()
class DistillatorRetinaNet(BaseDistillator):
    """
    """
    def __init__(self, cfg=None):
        """
        NOTE: this interface is experimental.

        Args:
            student: student detector
            teacher: dynamic teacher module
        """
        super().__init__(cfg)
        self.flag_seg_map = cfg.MODEL.DISTILLATOR.LABEL_ENCODER.LOAD_LABELMAP


    def forward(self, batched_inputs, **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            # student loss
            losses, r_features, features, images, gt_labels_boxes = self.forward_student(batched_inputs)
            # dynamic teacher loss
            losses_tea, _, features_tea, batchified_inside_masks, inst_labels = self.forward_teacher(batched_inputs, images=images,
                    r_features=r_features, features=features, gt_labels_boxes=gt_labels_boxes)
            # distill loss
            losses_distill = self.distill_loss({'stu': features, 'tea': features_tea}, images, batched_inputs, batchified_inside_masks, inst_labels)

            losses.update(losses_tea)
            #losses.update(losses_contrast)
            losses.update(losses_distill)
            return losses
        else:
            #anchors, pred_logits, pred_anchor_deltas, tower_feature_stu = self.predict(features)
            processed_results, r_features, features, images = self.forward_student(batched_inputs)

            anchors, pred_logits, pred_anchor_deltas = self.student.predict([features[f] for f in self.student.head_in_features])
            if kwargs.get('eval_teacher', False):
                # dynamic teacher
                features_tea, _, _ = self.teacher((batched_inputs, images, r_features, features))
                if isinstance(features_tea, dict):
                    features_tea = [features_tea[f] for f in self.student.head_in_features]
                # use student's head
                anchors, pred_logits, pred_anchor_deltas = self.student.predict(features_tea)
            # reuse test api
            results = self.student.inference(
                anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            processed_results = self.student.get_processed_results(results, batched_inputs, images)
            return processed_results

    def forward_student(self, batched_inputs, **kwargs):
        if self.training:
            losses, r_features, features, images, gt_labels_boxes = self.student(batched_inputs)
            return losses, r_features, features, images, gt_labels_boxes
        else:
            processed_results, r_features, features, images = self.student(batched_inputs)
            return processed_results, r_features, features, images

    def forward_teacher(self, batched_inputs, **kwargs):
        images = kwargs['images']
        r_features = kwargs['r_features']
        features = kwargs['features']
        gt_labels, gt_boxes = kwargs['gt_labels_boxes']
        # dynamic teacher have no fpn, thus None
        r_features_tea = None
        features_tea, inst_labels, batchified_inside_masks \
                = self.teacher((batched_inputs, images, r_features, features))

        # predict use student's head
        anchors, pred_logits_tea, pred_anchor_deltas_tea = self.student.predict(
                [features_tea[f] for f in self.student.head_in_features])
        # reuse loss function
        losses_tea = self.student.losses(
            anchors, pred_logits_tea, gt_labels, pred_anchor_deltas_tea, gt_boxes)
        losses_tea = {k+'.tea': v for k, v in losses_tea.items()}

        return losses_tea, r_features_tea, features_tea, batchified_inside_masks, inst_labels


@META_ARCH_REGISTRY.register()
class DistillatorGeneralizedRCNN(BaseDistillator):
    """
    """
    def __init__(self, cfg=None):
        """
        NOTE: this interface is experimental.

        Args:
            student: student detector
            teacher: dynamic teacher module
        """
        super().__init__(cfg)
        self.flag_seg_map = cfg.MODEL.DISTILLATOR.LABEL_ENCODER.LOAD_LABELMAP


    def forward(self, batched_inputs, **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            # student loss
            losses, r_features, features, images, gt_labels_boxes = self.forward_student(batched_inputs)
            # dynamic teacher loss
            losses_tea, _, features_tea, batchified_inside_masks, inst_labels = self.forward_teacher(batched_inputs, images=images,
                    r_features=r_features, features=features, gt_labels_boxes=gt_labels_boxes)
            # distill loss
            losses_distill = self.distill_loss({'stu': features, 'tea': features_tea}, images, batched_inputs, batchified_inside_masks, inst_labels)

            losses.update(losses_tea)
            #losses.update(losses_contrast)
            losses.update(losses_distill)
            return losses
        else:
            #anchors, pred_logits, pred_anchor_deltas, tower_feature_stu = self.predict(features)
            processed_results, r_features, features, images = self.forward_student(batched_inputs)

            if kwargs.get('eval_teacher', False):
                # dynamic teacher
                features_tea, _, _ = self.teacher((batched_inputs, images, r_features, features))
                # use student's head
                return self.student.inference(batched_inputs, features=features_tea)[0]
            # reuse test api
            return processed_results

    def forward_student(self, batched_inputs, **kwargs):
        if self.training:
            losses, r_features, features, images, gt_labels_boxes = self.student(batched_inputs)
            return losses, r_features, features, images, gt_labels_boxes
        else:
            processed_results, r_features, features, images = self.student(batched_inputs)
            return processed_results, r_features, features, images

    def forward_teacher(self, batched_inputs, **kwargs):
        images = kwargs['images']
        r_features = kwargs['r_features']
        features = kwargs['features']
        gt_instances = kwargs['gt_labels_boxes']
        # dynamic teacher have no fpn, thus None
        r_features_tea = None
        features_tea, inst_labels, batchified_inside_masks \
                = self.teacher((batched_inputs, images, r_features, features))

        # reuse loss function
        losses_tea = self.student.predict(features_tea, images, gt_instances, batched_inputs)
        losses_tea = {k+'.tea': v for k, v in losses_tea.items()}

        return losses_tea, r_features_tea, features_tea, batchified_inside_masks, inst_labels


@META_ARCH_REGISTRY.register()
class DistillatorFCOS(BaseDistillator):
    """
    """
    def __init__(self, cfg=None):
        """
        NOTE: this interface is experimental.

        Args:
            student: student detector
            teacher: dynamic teacher module
        """
        super().__init__(cfg)
        self.flag_seg_map = cfg.MODEL.DISTILLATOR.LABEL_ENCODER.LOAD_LABELMAP

    def forward(self, batched_inputs, **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            # student loss
            losses, r_features, features, images, gt_targets = self.forward_student(batched_inputs)
            # dynamic teacher loss
            losses_tea, _, features_tea, batchified_inside_masks, inst_labels = self.forward_teacher(batched_inputs, images=images,
                    r_features=r_features, features=features, gt_targets=gt_targets)
            # distill loss
            losses_distill = self.distill_loss({'stu': features, 'tea': features_tea}, images, batched_inputs, batchified_inside_masks, inst_labels)

            losses.update(losses_tea)
            #losses.update(losses_contrast)
            losses.update(losses_distill)
            return losses
        else:
            #anchors, pred_logits, pred_anchor_deltas, tower_feature_stu = self.predict(features)
            processed_results, r_features, features, images = self.forward_student(batched_inputs)

            #anchors, pred_logits, pred_anchor_deltas = self.student.predict([features[f] for f in self.student.head_in_features])

            shifts, box_cls, box_delta, box_center = self.student.predict([features[f] for f in self.student.in_features])


            if kwargs.get('eval_teacher', False):
                # dynamic teacher
                features_tea, _, _ = self.teacher((batched_inputs, images, r_features, features))
                if isinstance(features_tea, dict):
                    features_tea = [features_tea[f] for f in self.student.in_features]
                # use student's head
                shifts, box_cls, box_delta, box_center = self.student.predict(features_tea)
            # reuse test api
            results = self.student.inference(
                    box_cls, box_delta, box_center, shifts, images)
            processed_results = self.student.get_processed_results(results, batched_inputs, images)
            return processed_results

    def forward_student(self, batched_inputs, **kwargs):
        if self.training:
            losses, r_features, features, images, gt_targets = self.student(batched_inputs)
            return losses, r_features, features, images, gt_targets
        else:
            processed_results, r_features, features, images = self.student(batched_inputs)
            return processed_results, r_features, features, images

    def forward_teacher(self, batched_inputs, **kwargs):
        images = kwargs['images']
        r_features = kwargs['r_features']
        features = kwargs['features']
        gt_classes, gt_shifts_reg_deltas, gt_centerness = kwargs['gt_targets']
        # dynamic teacher have no fpn, thus None
        r_features_tea = None
        features_tea, inst_labels, batchified_inside_masks \
                = self.teacher((batched_inputs, images, r_features, features))

        # predict use student's head
        shifts, box_cls, box_delta, box_center = self.student.predict(
                [features_tea[f] for f in self.student.in_features])
        # reuse loss function

        losses_tea = self.student.losses(gt_classes, gt_shifts_reg_deltas,
                gt_centerness, box_cls, box_delta, box_center)

        losses_tea = {k+'.tea': v for k, v in losses_tea.items()}

        return losses_tea, r_features_tea, features_tea, batchified_inside_masks, inst_labels

@META_ARCH_REGISTRY.register()
class DistillatorPOTO(BaseDistillator):
    """
    """
    def __init__(self, cfg=None):
        """
        NOTE: this interface is experimental.

        Args:
            student: student detector
            teacher: dynamic teacher module
        """
        super().__init__(cfg)
        self.flag_seg_map = cfg.MODEL.DISTILLATOR.LABEL_ENCODER.LOAD_LABELMAP


    def forward(self, batched_inputs, **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            # student loss
            losses, r_features, features, images, gt_targets = self.forward_student(batched_inputs)
            # dynamic teacher loss
            losses_tea, _, features_tea, batchified_inside_masks, inst_labels = self.forward_teacher(batched_inputs, images=images,
                    r_features=r_features, features=features, gt_targets=gt_targets)
            # distill loss
            losses_distill = self.distill_loss({'stu': features, 'tea': features_tea}, images, batched_inputs, batchified_inside_masks, inst_labels)

            losses.update(losses_tea)
            #losses.update(losses_contrast)
            losses.update(losses_distill)
            return losses
        else:
            #anchors, pred_logits, pred_anchor_deltas, tower_feature_stu = self.predict(features)
            processed_results, r_features, features, images = self.forward_student(batched_inputs)

            #anchors, pred_logits, pred_anchor_deltas = self.student.predict([features[f] for f in self.student.head_in_features])

            shifts, box_cls, box_delta = self.student.predict([features[f] for f in self.student.in_features])


            if kwargs.get('eval_teacher', False):
                # dynamic teacher
                features_tea, _, _ = self.teacher((batched_inputs, images, r_features, features))
                if isinstance(features_tea, dict):
                    features_tea = [features_tea[f] for f in self.student.in_features]
                # use student's head
                shifts, box_cls, box_delta = self.student.predict(features_tea)
            # reuse test api
            results = self.student.inference(box_cls, box_delta, shifts, images)
            processed_results = self.student.get_processed_results(results, batched_inputs, images)
            return processed_results

    def forward_student(self, batched_inputs, **kwargs):
        if self.training:
            losses, r_features, features, images, gt_targets = self.student(batched_inputs)
            return losses, r_features, features, images, gt_targets
        else:
            processed_results, r_features, features, images = self.student(batched_inputs)
            return processed_results, r_features, features, images

    def forward_teacher(self, batched_inputs, **kwargs):
        images = kwargs['images']
        r_features = kwargs['r_features']
        features = kwargs['features']
        gt_classes, gt_shifts_reg_deltas = kwargs['gt_targets']
        # dynamic teacher have no fpn, thus None
        r_features_tea = None
        features_tea, inst_labels, batchified_inside_masks \
                = self.teacher((batched_inputs, images, r_features, features))

        # predict use student's head
        shifts, box_cls, box_delta = self.student.predict(
                [features_tea[f] for f in self.student.in_features])
        # reuse loss function

        losses_tea = self.student.losses(gt_classes, gt_shifts_reg_deltas,
                box_cls, box_delta)

        losses_tea = {k+'.tea': v for k, v in losses_tea.items()}

        return losses_tea, r_features_tea, features_tea, batchified_inside_masks, inst_labels

@META_ARCH_REGISTRY.register()
class DistillatorATSS(BaseDistillator):
    """
    """
    def __init__(self, cfg=None):
        """
        NOTE: this interface is experimental.

        Args:
            student: student detector
            teacher: dynamic teacher module
        """
        super().__init__(cfg)
        self.flag_seg_map = cfg.MODEL.DISTILLATOR.LABEL_ENCODER.LOAD_LABELMAP


    def forward(self, batched_inputs, **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            # student loss
            losses, r_features, features, images, gt_targets = self.forward_student(batched_inputs)
            # dynamic teacher loss
            losses_tea, _, features_tea, batchified_inside_masks, inst_labels = self.forward_teacher(batched_inputs, images=images,
                    r_features=r_features, features=features, gt_targets=gt_targets)
            # distill loss
            losses_distill = self.distill_loss({'stu': features, 'tea': features_tea}, images, batched_inputs, batchified_inside_masks, inst_labels)

            losses.update(losses_tea)
            #losses.update(losses_contrast)
            losses.update(losses_distill)
            return losses
        else:
            #anchors, pred_logits, pred_anchor_deltas, tower_feature_stu = self.predict(features)
            processed_results, r_features, features, images = self.forward_student(batched_inputs)

            #anchors, pred_logits, pred_anchor_deltas = self.student.predict([features[f] for f in self.student.head_in_features])

            shifts, box_cls, box_delta, box_center = self.student.predict([features[f] for f in self.student.in_features])


            if kwargs.get('eval_teacher', False):
                # dynamic teacher
                features_tea, _, _ = self.teacher((batched_inputs, images, r_features, features))
                if isinstance(features_tea, dict):
                    features_tea = [features_tea[f] for f in self.student.in_features]
                # use student's head
                shifts, box_cls, box_delta, box_center = self.student.predict(features_tea)
            # reuse test api
            results = self.student.inference(
                    box_cls, box_delta, box_center, shifts, images)
            processed_results = self.student.get_processed_results(results, batched_inputs, images)
            return processed_results

    def forward_student(self, batched_inputs, **kwargs):
        if self.training:
            losses, r_features, features, images, gt_targets = self.student(batched_inputs)
            return losses, r_features, features, images, gt_targets
        else:
            processed_results, r_features, features, images = self.student(batched_inputs)
            return processed_results, r_features, features, images

    def forward_teacher(self, batched_inputs, **kwargs):
        images = kwargs['images']
        r_features = kwargs['r_features']
        features = kwargs['features']
        gt_classes, gt_shifts_reg_deltas, gt_centerness = kwargs['gt_targets']
        # dynamic teacher have no fpn, thus None
        r_features_tea = None
        features_tea, inst_labels, batchified_inside_masks \
                = self.teacher((batched_inputs, images, r_features, features))

        # predict use student's head
        shifts, box_cls, box_delta, box_center = self.student.predict(
                [features_tea[f] for f in self.student.in_features])
        # reuse loss function

        losses_tea = self.student.losses(gt_classes, gt_shifts_reg_deltas,
                gt_centerness, box_cls, box_delta, box_center)

        losses_tea = {k+'.tea': v for k, v in losses_tea.items()}

        return losses_tea, r_features_tea, features_tea, batchified_inside_masks, inst_labels
