# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
# Modified from cvpods (https://github.com/Megvii-BaseDetection/cvpods)
# Copyright (c) Megvii, Inc. All Rights Reserved
# ------------------------------------------------------------------------------
import logging

from cvpods.modeling.anchor_generator import ShiftGenerator
import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances, Boxes, pairwise_iou

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import iou_loss, sigmoid_focal_loss_jit
from cvpods.utils import comm, log_first_n
from scipy.optimize import linear_sum_assignment

from .scale import Scale

def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def permute_all_cls_and_box_to_N_HWA_K_and_concat_normal(box_cls,
                                                  box_delta,
                                                  num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


class POTO(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.reg_weight = cfg.MODEL.POTO.REG_WEIGHT
        # Inference parameters:
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = build_backbone(cfg)
        self.backbone_shape = self.backbone.output_shape()
        self.feature_shapes = [self.backbone_shape[f] for f in cfg.MODEL.FCOS.IN_FEATURES]
        self.head = POTOHead(cfg, self.feature_shapes)
        self.shift_generator = ShiftGenerator(cfg, self.feature_shapes)


        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.poto_alpha = cfg.MODEL.POTO.ALPHA
        self.center_sampling_radius = cfg.MODEL.POTO.CENTER_SAMPLING_RADIUS

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, eval_teacher=False, requires_raw=False, forward_only=False):
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
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10)
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        r_features_stu = self.backbone(images.tensor)
        features_stu = self.fpn_head(r_features_stu)
        features = [features_stu[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        shifts = self.shift_generator(features)

        if self.training:
            if forward_only:
                self.forward_only = True
            else:
                self.forward_only = False

            gt_classes, gt_shifts_reg_deltas = self.get_ground_truth(
                shifts, gt_instances, box_cls, box_delta)
            losses = self.losses(gt_classes, gt_shifts_reg_deltas,
                               box_cls, box_delta)

            losses_tea, features_tea, fpn_teas, gt_cover_mask = self.forward_teacher(batched_inputs, images, r_features_stu, features_stu, gt_instances)
            losses.update(losses_tea)
            losses["loss_distill"] = self.distill(
                    features_stu, fpn_teas, images, batched_inputs, gt_cover_mask)

            if requires_raw:
                return losses, fpn_teas, features_stu

            return losses
        else:
            if eval_teacher:
                features_tea_, trans_matrice_desc, trans_matrice_feat, gt_cover_mask = self.encoder((batched_inputs, images, r_features_stu, features_stu))

                features_tea = self.fpn_tea(features_tea_)
                features = [features_tea[f] for f in self.in_features]
                box_cls, box_delta = self.head(features)
                shifts = self.shift_generator(features)

            results = self.inference(box_cls, box_delta, shifts,
                                     images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results


    def _forward_teas(self, batched_inputs, images, r_features, features, gts):
        if self.forward_only:
            for p in self.encoder.parameters():
                p.grad = None
            for p in self.fpn_tea.parameters():
                p.grad = None
            # to get rid of weight decay

            with torch.no_grad():
                r_features_tea = self.encoder(
                    (batched_inputs, images, r_features, features))[0]
                features_tea = self.fpn_tea(r_features_tea)
                return {}, r_features_tea, features_tea

        if self.forbid_teacher_grad:
            set_requires_grad(self.head, False)

        gt_instances = gts
        # forward teacher (with FPN)
        fpn_raw, trans_matrice_desc, trans_matrice_feat, gt_cover_mask = self.encoder((batched_inputs, images, r_features, features))
        fpn_feats = self.fpn_tea(fpn_raw)
        features = [fpn_feats[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        shifts = self.shift_generator(features)

        gt_classes, gt_shifts_reg_deltas = self.get_ground_truth(
                shifts, gt_instances, box_cls, box_delta)

        losses = self.losses(gt_classes, gt_shifts_reg_deltas,
                               box_cls, box_delta)
        losses = {k+'.tea': v for k, v in losses.items()}

        if self.forbid_teacher_grad:
            set_requires_grad(self.head, True)

        #return losses, fpn_raw, fpn_feats
        return losses, fpn_raw, fpn_feats, gt_cover_mask


    def losses(self, gt_classes, gt_shifts_deltas, pred_class_logits,
               pred_shift_deltas):
        """
        Args:
            For `gt_classes` and `gt_shifts_deltas` parameters, see
                :meth:`FCOS.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits` and `pred_shift_deltas`, see
                :meth:`FCOSHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_shift_deltas = \
            permute_all_cls_and_box_to_N_HWA_K_and_concat_normal(
                pred_class_logits, pred_shift_deltas, self.num_classes
            )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
        ) / max(1.0, num_foreground) * self.reg_weight

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
        }

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, box_cls, box_delta):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
        """
        gt_classes = []
        gt_shifts_deltas = []

        box_cls = torch.cat([permute_to_N_HWA_K(x, self.num_classes) for x in box_cls], dim=1)
        box_delta = torch.cat([permute_to_N_HWA_K(x, 4) for x in box_delta], dim=1)
        box_cls = box_cls.sigmoid_()

        num_fg = 0
        num_gt = 0

        for shifts_per_image, targets_per_image, box_cls_per_image, box_delta_per_image in zip(
                shifts, targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            prob = box_cls_per_image[:, targets_per_image.gt_classes].t()
            boxes = self.shift2box_transform.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            quality[~is_in_boxes] = -1

            gt_idxs, shift_idxs = linear_sum_assignment(quality.cpu().numpy(), maximize=True)

            num_fg += len(shift_idxs)
            num_gt += len(targets_per_image)

            gt_classes_i = shifts_over_all_feature_maps.new_full(
                (len(shifts_over_all_feature_maps),), self.num_classes, dtype=torch.long
            )
            gt_shifts_reg_deltas_i = shifts_over_all_feature_maps.new_zeros(
                len(shifts_over_all_feature_maps), 4
            )
            if len(targets_per_image) > 0:
                # ground truth classes
                gt_classes_i[shift_idxs] = targets_per_image.gt_classes[gt_idxs]
                # ground truth box regression
                gt_shifts_reg_deltas_i[shift_idxs] = self.shift2box_transform.get_deltas(
                    shifts_over_all_feature_maps[shift_idxs], gt_boxes[gt_idxs].tensor
                )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)

        #get_event_storage().put_scalar("num_fg_per_gt", num_fg / num_gt)

        return torch.stack(gt_classes), torch.stack(gt_shifts_deltas)

    def inference(self, box_cls, box_delta, shifts, images):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`FCOSHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, shifts_per_image,
                tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, shifts, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, shifts_i in zip(box_cls, box_delta, shifts):
            # (HxWxK,)
            box_cls_i = box_cls_i.sigmoid_().flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        if self.nms_type == 'null':
            # strategies above (e.g. topk_candidates and score_threshold) are
            # useless for POTO, just keep them for debug and analysis
            keep = scores_all.argsort(descending=True)
        else:
            keep = generalized_batched_nms(
                boxes_all, scores_all, class_idxs_all,
                self.nms_threshold, nms_type=self.nms_type
            )
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    #def _inference_for_ms_test(self, batched_inputs):
    #    """
    #    function used for multiscale test, will be refactor in the future.
    #    The same input with `forward` function.
    #    """
    #    assert not self.training, "inference mode with training=True"
    #    assert len(batched_inputs) == 1, "inference image number > 1"
    #    images = self.preprocess_image(batched_inputs)

    #    features = self.backbone(images.tensor)
    #    features = [features[f] for f in self.in_features]
    #    box_cls, box_delta = self.head(features)
    #    shifts = self.shift_generator(features)

    #    results = self.inference(box_cls, box_delta, shifts, images)
    #    for results_per_image, input_per_image, image_size in zip(
    #            results, batched_inputs, images.image_sizes
    #    ):
    #        height = input_per_image.get("height", image_size[0])
    #        width = input_per_image.get("width", image_size[1])
    #        processed_results = detector_postprocess(results_per_image, height, width)
    #    return processed_results


class POTOHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        num_shifts = ShiftGenerator(cfg, input_shape).num_cell_shifts
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        # fmt: on
        assert len(set(num_shifts)) == 1, "using differenct num_shifts value is not supported"
        num_shifts = num_shifts[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_shifts * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_shifts * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for level, feature in enumerate(features):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[level])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg
