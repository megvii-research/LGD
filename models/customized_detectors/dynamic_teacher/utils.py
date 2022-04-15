# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# The api: `MASKS.polygons_to_bitmask` comes from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from collections import namedtuple
from functools import reduce
import detectron2.structures.masks as MASKS

RangeScaler = namedtuple('RangeScaler', 'a b Min Max')
resolution = namedtuple('resolution', 'h w')

def range_scaling(rs, x):
    '''
    Inputs:
        rs: RangerScaler
        x: tensor with the same range, e.g. [0, 1]
    Outputs:
        scaled output tensor from range [Min, Max] to range [a, b]
    '''
    return (rs.b-rs.a) / (rs.Max - rs.Min) * (x - rs.Min) + rs.a

def x1y1wh_to_x1y1x2y2(bboxes):
    '''
    Input:
        bboxes: (N, 4)
    Output:
        new_bboxes: (N, 4)
    '''
    x1s = bboxes[:, 0]
    y1s = bboxes[:, 1]
    x2s = x1s + bboxes[:, 2] - 1.0
    y2s = y1s + bboxes[:, 3] - 1.0
    new_bboxes = torch.stack([x1s, y1s, x2s, y2s], dim=1)
    return new_bboxes

def clamp_x1y1x2y2(bboxes, img_h, img_w):
    '''
    Input:
        bboxes of x1y1x2y2 mode, (N, 4), supposed the bboxes coors start from 0
    Output:
        clamped bboxes
    '''
    x1 = torch.clamp(bboxes[:, 0], min=0, max=img_w-1)
    x2 = torch.clamp(bboxes[:, 2], min=0, max=img_w-1)
    y1 = torch.clamp(bboxes[:, 1], min=0, max=img_h-1)
    y2 = torch.clamp(bboxes[:, 3], min=0, max=img_h-1)
    return torch.stack([x1, y1, x2, y2], dim=1)

@torch.no_grad()
def get_inside_gt_mask(boxlist, src, dst, device):
    '''
    Input:
        boxlist: box list of (N, 4) of a single image with N objs.
                 The coordinate count from 0, guaranteed in x1y1x2y2 format (by tracing the source) and clamped
        src: resolution tuple, encompassing the boxlist
        dst: resolution tuple, for the target image space
    Output:
        inside_gt_mask:  (N, dst.h * dst.w)
    '''
    # (N, 4), box coordinates in target resolution scale
    box_tensor = torch.FloatTensor(boxlist).to(device)
    #box_tensor = clamp_x1y1x2y2(box_tensor, src.h, src.w)
    r_h, r_w = dst.h / src.h, dst.w / src.w

    #box_tensor[:, [0,2]] *= r_w
    #box_tensor[:, [1,3]] *= r_h
    box_tensor[:, [0,2]] = box_tensor[:, [0,2]] * r_w
    box_tensor[:, [1,3]] = box_tensor[:, [1,3]] * r_h

    # (N, 2)
    xc = (box_tensor[:, 0] + box_tensor[:, 2]) * 0.5
    yc = (box_tensor[:, 1] + box_tensor[:, 3]) * 0.5
    centers = torch.stack([yc, xc], dim=1)

    w_ = box_tensor[:, 2] - box_tensor[:, 0]
    h_ = box_tensor[:, 3] - box_tensor[:, 1]
    scales = torch.stack([h_, w_], dim=1)
    # (dst.h, dst.w)
    ys, xs = torch.meshgrid(torch.arange(dst.h).to(device), torch.arange(dst.w).to(device))
    # (2, dst.h, dst.w)
    coors = torch.stack([ys, xs], dim=0)
    # (N, 2, dst.h, dst.w)
    dist = torch.abs(centers[:, :, None, None] - coors[None, :, :, :]) / scales[:, :, None, None]
    inside_gt_mask = ((dist <= 0.5).all(1)).flatten(start_dim=1).float()
    return inside_gt_mask


@torch.no_grad()
def get_segmask_inside_gt(hws, batched_inputs, src, device, add_bg_box=False):
    # transform polymask to seg map style inputs
    # this is for detectron 2, mask_color_encode should be very similar to this version

    images = [x["image"] for x in batched_inputs]
    boxlists = [x["instances"] for x in batched_inputs]
    # H, W = src.h, src.w

    scales_imgs_per_ins = []
    for i in range(len(boxlists)):
        label_idx = 0
        boxlist = boxlists[i]
        labels = boxlist.gt_classes # 0 for background
        C, H, W = images[i].shape
        box_size = max(len(labels) + (1 if add_bg_box else 0), 1)
        if len(labels) > 0:
            masks = boxlist.gt_masks
            target_img = torch.zeros([box_size, H * W], device=device)

            for label_idx, mask in enumerate(masks):
                mask = MASKS.polygons_to_bitmask(mask, H, W).reshape(-1) # there could be overlap, but we just ignore them
                target_img[label_idx, mask] = 1.0 # fill in index
        else:
            target_img = torch.zeros([box_size, H * W], device=device)
            label_idx = -1

        if add_bg_box:
            target_img[label_idx + 1, :] = 1.0 # fill in index

        target_img = target_img.reshape(1, box_size, H, W).float()

        target_img = F.pad(target_img, (0, src.w-W, 0, src.h-H))

        imgs_per_scales_per_batch = []
        for h, w in hws:
            imgs_per_scales_per_batch.append(F.interpolate(target_img, size=(h, w), mode='nearest').squeeze(0).flatten(start_dim=1))

        scales_imgs_per_ins.append(imgs_per_scales_per_batch)

    return [x for x in zip(*scales_imgs_per_ins)]
