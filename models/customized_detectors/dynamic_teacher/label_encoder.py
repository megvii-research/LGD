# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_transformer import STN
from .utils import (range_scaling, x1y1wh_to_x1y1x2y2, clamp_x1y1x2y2, get_inside_gt_mask, RangeScaler)
from copy import deepcopy


@torch.no_grad()
def box_descriptor_encode(boxListObj_list, img_h, img_w, num_classes, category_format, box_format='x1y1x2y2', add_context_box=False, add_mask=False):
    '''
    Inputs:
        boxListObj_list: list of BoxList obj represent the batch annotations. Each obj contains information of that image
        num_classes: number of foreground classes
        box_format: by default process x1y1x2y2 form only, if input x1y1wh then should apply convert first
    Output:
        batch_descriptors: list of (Ni, k), k denote the descriptor length. The list should be of length B (batch-size)
                           Normally, k = 4+1 or 4+num_classes
        boxlists: nested list of B x (Ni, 4), n might vary from different image. i = 1, ..., B
    '''
    if category_format == 'norm_classes':
        k = 4 + 1
    elif category_format == 'one_hot':
        k = 4 + num_classes
    else:
        k = 11 + num_classes

    if add_mask:
        k += 49

    assert box_format in ('x1y1x2y2', 'x1y1wh')
    rs = RangeScaler(a=-1, b=1, Min=0, Max=1)

    batch_descriptors = list()
    boxlists = list()
    instance_labels = list()

    device = boxListObj_list[0].gt_boxes.device
    for boxlistObj in boxListObj_list:
        nr_boxes_cur_img = len(boxlistObj)

        #assert nr_boxes_cur_img > 0
        # (Ni, 4)
        #import ipdb
        #ipdb.set_trace()
        #assert False
        #TODO: note that there might be the cases that in the inference time
        # there are some val images do not have any ground-truth objects which
        # will make the bboxes and labels empty. here I just use special single
        # point box instead to represent them.
        # Note that this also happens in beforehand teacher too. How does the logic
        # they handle this
        # But it seems no-gt case will not happen during training
        if nr_boxes_cur_img > 0:
            bboxes = boxlistObj.gt_boxes.tensor.reshape(nr_boxes_cur_img, 4)
            labels = boxlistObj.gt_classes.reshape((nr_boxes_cur_img, 1)) # start from 0
            if add_mask:
                masks = boxlistObj.gt_masks.crop_and_resize(
                            boxlistObj.gt_boxes.tensor, 7
                        ).reshape((nr_boxes_cur_img, 49))
        else:
            bboxes = torch.tensor([0.0, 0.0, 1.0, 1.0]).reshape(1, 4).to(device)
            labels = torch.zeros((1, 1)).to(device)

            if add_mask:
                masks = torch.zeros(1, 49).to(device)

        # transfer to x1y1x2y2
        if box_format == 'x1y1wh':
            bboxes = x1y1wh_to_x1y1x2y2(bboxes)

        if add_context_box and nr_boxes_cur_img > 0:
            bboxes = torch.cat((bboxes, torch.tensor([[0.0, 0.0, img_w, img_h]], device=device)), 0)
            nr_boxes_cur_img += 1

            if add_mask:
                masks = torch.cat((masks, torch.ones(1, 49).to(device)), 0)

        # clamp the box boundaries
        bboxes = clamp_x1y1x2y2(bboxes, img_h, img_w)

        boxlists.append(deepcopy(bboxes).tolist())

        # range from (0, 1)
        bboxes[:, [0,2]] /= img_w
        bboxes[:, [1,3]] /= img_h
        # range from (0, 1)
        if category_format == 'norm_classes':
           # (Ni, 1)
            category_embed = labels / num_classes
        elif category_format == 'one_hot':
           # (Ni, num_classes)
            # first let label start from 0 then scatter
            if nr_boxes_cur_img > 0:
                assert ((labels >= 0)*(labels<=num_classes-1)).all().item()
                category_embed = torch.zeros(nr_boxes_cur_img, num_classes).scatter_(1, labels, 1).to(device)
            else:
                #TODO: for the no gt case
                category_embed = torch.zeros(1, num_classes).to(device)
        else:
            raise ValueError('Unsupported class_descriptor mode: {} !'.format(category_format))
        box_descriptor = torch.cat([bboxes, category_embed], dim=1)
        if add_mask:
            box_descriptor = torch.cat([box_descriptor, masks], dim=1)
        assert torch.all((box_descriptor >= 0) * (box_descriptor <= 1)).item()
        #NOTE: the range_scaling of here could not be calibrated like that in the 'def calc_prior_channels'
        # since ground-truth not guarantee to be both the min and max border
        # range from (0, 1) to (-1, 1)
        box_descriptor = range_scaling(rs, box_descriptor)
        batch_descriptors.append(box_descriptor)
        instance_labels.append(labels.reshape(-1))
    return batch_descriptors, boxlists, instance_labels



class LabelEncoder(nn.Module):
    def __init__(self, category_format='norm_classes', box_format='x1y1x2y2',
                       nr_fg_classes=80, noise_std=0.0, add_context_box=False, parse_mask=False):
        '''
        category_format:
            Description of the obj (of the box or instance) class
                (1) 'norm_classes': class_tag / num_classes (class_tag >=1)
                (2) 'one_hot' : one hot representation of the object category label
        '''
        super(LabelEncoder, self).__init__()
        self.category_format = category_format
        self.box_format = box_format
        self.nr_fg_classes = nr_fg_classes
        self.R = 1
        self.noise_std = noise_std
        self.add_context_box = add_context_box

        if category_format == 'norm_classes':
            self.inp = 4+1
        elif category_format == 'one_hot':
            self.inp = 4+self.nr_fg_classes
        else:
            raise ValueError('category_format {} not supported yet !'.format(self.category_format))

        self.parse_mask = parse_mask
        if parse_mask:
            self.inp += 49

        self.stn_desc = STN(self.inp)
        self.stn_feat = STN(64)
        self.conv1 = nn.Conv1d(self.inp, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # fuse
        # global + local feat
        self.conv4 = nn.Conv1d(1088, 256, 1)

        self.bn1 = nn.LayerNorm([64, 1], elementwise_affine=False)
        self.bn2 = nn.LayerNorm([128, 1], elementwise_affine=False)
        self.bn3 = nn.LayerNorm([1024, 1], elementwise_affine=False)
        self.bn4 = nn.LayerNorm([256, 1], elementwise_affine=False)



    @torch.no_grad()
    def get_init_descriptors(self, batched_inputs, images):
        targets = [x['instances'] for x in batched_inputs]
        _, _, h, w = images.tensor.size()
        descriptors, boxlists, inst_labels = box_descriptor_encode(targets, h, w, self.nr_fg_classes, self.category_format, self.box_format, self.add_context_box, add_mask=self.parse_mask)
        return descriptors, {'h': h, 'w': w}, boxlists, inst_labels

    @torch.no_grad()
    def prepare_descriptors(self, x):
        batched_inputs, images, raw_outputs, fpn_outputs = x
        x, img_size_dict, boxlists, inst_labels = self.get_init_descriptors(batched_inputs, images)
        nr_gt_per_img = torch.LongTensor([ts.size()[0] for ts in x])
        return x, nr_gt_per_img, img_size_dict, boxlists, inst_labels

    @torch.no_grad()
    def repeat_and_compose(self, x):
        '''
        Input:
            x: same explanation as def forwards's
        Output:
            (T*R, k, 1)
        '''
        # (T, k, R)
        res = torch.cat(x, dim=0).unsqueeze(-1).repeat(1, 1, self.R)
        # (T*R, k, 1)
        res = res.permute(0, 2, 1).reshape(-1, self.inp, 1)

        res = res if self.noise_std == 0.0 else res + torch.normal(0, self.noise_std, size=res.shape)

        return res

    def hier_pool(self, x, nr_gt_per_img):
        '''
        Hierarchical pooling: first with the repeatitions, then within each image
        Input:
            x: (T*R, C, 1), feature
            nr_gt_per_img: (B,), number of ground-truth boxes each image.
                Note that T=N1+..._NB
        Output:
            x: (B, C, 1)
        '''
        # within the repeatitions
        T = nr_gt_per_img.sum().item()
        # (T, C, 1)
        x = x.reshape(T, self.R, -1, 1).mean(dim=1)
        # [..., (Ni, C, 1), ...]
        x_ = x.split(nr_gt_per_img.tolist(), dim=0)
        x_ = [feat_per_img.max(dim=0)[0] for feat_per_img in x_]
        x_ = torch.stack(x_, dim=0)
        return x_


    def forward(self, x0):
        '''
        Input:
            x0: input information: a tuple of (batched_inputs, images, r_features, features)
        Output:
            x: output tensor with shape (T, 256)
            trans_matrice_desc: (T*R, k, k)
            trans_matrice_feat: (T*R, 64, 64)
            boxlists: nested list of B x (Ni, 4), n might vary from different image. i = 1, ..., B
                     The coordinate count from 0, guaranteed in x1y1x2y2 format (by tracing the source) and clamped
        '''
        # x is the batch descriptors
        # list of (Ni, k), k denote the descriptor length. list length B
        x, nr_gt_per_img, img_size_dict, boxlists, inst_labels = self.prepare_descriptors(x0)

        k = self.inp
        # compose x in to (T*R, k, 1)
        x = self.repeat_and_compose(x)

        device = x0[-1]['p3'].device
        x = x.to(device)

        # (T*R, k, k)
        trans_matrice_desc = self.stn_desc(x)
        # descriptor transform
        x = torch.bmm(x.permute(0, 2, 1), trans_matrice_desc).permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        # (T*R, 64, 64)
        trans_matrice_feat = self.stn_feat(x)
        # (T*R, 64, 1)
        x_feat_trans = torch.bmm(x.permute(0, 2, 1), trans_matrice_feat).permute(0, 2, 1)

        x = F.relu(self.bn2(self.conv2(x_feat_trans)))

        # (T*R, C, 1)
        x = F.relu(self.bn3(self.conv3(x)))

        TR = x.size()[0]
        # (B, C, 1)
        x_g = self.hier_pool(x, nr_gt_per_img)
        # spread x_g (B, C, 1) onto (T*R, C, 1)
        bs = x_g.size()[0]
        C = x_g.size()[1]
        # list of (Ni, C, 1)
        x_g_remap = [x_gt_per_img.reshape(1, -1, 1).repeat(nr_gt_cur_img, 1, 1) for x_gt_per_img, nr_gt_cur_img in zip(x_g, nr_gt_per_img.tolist())]
        # (T*R, C, 1)
        x_g_remap = torch.cat(x_g_remap, dim=0).unsqueeze(1).repeat(1, self.R, 1, 1).view(TR, -1, 1)

        # (T*R, 64+C, 1)
        x_cat = torch.cat([x_feat_trans, x_g_remap], dim=1)

        # (T*R, 256, 1)
        x = F.relu(self.bn4(self.conv4(x_cat)))

        # Average
        # (T, 256)
        x = x.reshape(-1, self.R, 256, 1).mean(dim=1).squeeze(-1)

        return x, trans_matrice_desc, trans_matrice_feat, boxlists, img_size_dict, inst_labels
