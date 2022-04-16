# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import CUSTOMIZED_DETECTORS_REGISTRY
from .label_encoder import LabelEncoder
from .utils import get_segmask_inside_gt, get_inside_gt_mask, resolution
from functools import reduce
from .layers import get_norm, get_MLP, get_CONVS


#NOTE(peizhen): the dynamic teacher parameterize the (1) label encoder (2) inter-object relation adapter and (3) intra-object knowledge mapper and also the (4) parameter-free appearance encoder

@CUSTOMIZED_DETECTORS_REGISTRY.register()
class DynamicTeacher(nn.Module):
    """
    dynamic teacher input with set embedding
    In order to exploit and share the decoder with the traditional student network
    There might be the usage of remapping function (if sparse interaction) for rendering
    """
    def __init__(self, cfg):
        super().__init__()
        self.nr_fpn_channels = cfg.MODEL.FPN.OUT_CHANNELS # 256 normally
        self.num_classes = cfg.NUM_CLASSES # 80 normally

        assert self.nr_fpn_channels == 256
        assert self.num_classes == 80


        self.interact_pattern = cfg.MODEL.DISTILLATOR.TEACHER.INTERACT_PATTERN

        # strides: list of stride information for each fpn levels, e.g., (1/8, 1/16, 1/32, 1/64, 1/128)
        self.strides = cfg.MODEL.RECIPROCAL_FPN_STRIDES

        # box_format: 'x1y1x2y2' or 'x1y1wh'
        self.box_format =      cfg.MODEL.DISTILLATOR.LABEL_ENCODER.BOX_FORMAT
        self.category_format = cfg.MODEL.DISTILLATOR.LABEL_ENCODER.CATEGORY_FORMAT
        self.use_seg_map =     cfg.MODEL.DISTILLATOR.LABEL_ENCODER.LOAD_LABELMAP

        self.add_context_box = cfg.MODEL.DISTILLATOR.TEACHER.ADD_CONTEXT_BOX


        self.detach_appearance_embed = cfg.MODEL.DISTILLATOR.TEACHER.DETACH_APPEARANCE_EMBED

        self.label_encoder_ = LabelEncoder(
                    category_format=self.category_format,
                    box_format=self.box_format,
                    nr_fg_classes=self.num_classes,
                    add_context_box=self.add_context_box, parse_mask=self.use_seg_map)

        self.render_divide_occurence = False
        self.affine_flag = False

        self.canoni_proj_1D =    get_MLP(1, self.nr_fpn_channels, has_norm=True, has_relu=True, affine_flag=self.affine_flag)
        self.student_proj_2D = get_CONVS(1, self.nr_fpn_channels, has_norm=True, has_relu=True, nr_groups=1, affine_flag=self.affine_flag) # GN with nr_groups=1 indicate LN

        # After interaction

        self.local_inst_proj_2D = nn.Conv2d(self.nr_fpn_channels, self.nr_fpn_channels, 3, 1, 1)

        self.global_ctx_proj_1D = nn.Linear(self.nr_fpn_channels, self.nr_fpn_channels)

        self.local_inst_proj_1D = nn.Linear(self.nr_fpn_channels, self.nr_fpn_channels)

        self.refinement_module = nn.Sequential(
                nn.Conv2d(self.nr_fpn_channels, self.nr_fpn_channels, 3, 1, 1),
                get_norm(self.nr_fpn_channels, 1, self.affine_flag), nn.ReLU(),
                nn.Conv2d(self.nr_fpn_channels, self.nr_fpn_channels, 3, 1, 1),
                get_norm(self.nr_fpn_channels, 1, self.affine_flag), nn.ReLU(),
                nn.Conv2d(self.nr_fpn_channels, self.nr_fpn_channels, 3, 1, 1),
                get_norm(self.nr_fpn_channels, 1, self.affine_flag))


        self.nr_transformer_heads = cfg.MODEL.DISTILLATOR.TEACHER.NR_TRANSFORMER_HEADS

        self.multi_head_attn = nn.MultiheadAttention(self.nr_fpn_channels, self.nr_transformer_heads)


    def aggregate_per_level(self, stu_feat_cur_level, inst_masks):
        '''
        Inputs:
            stu_feat_cur_level: (B, C, Hi*Wi)
            inst_masks: list: B x [Ni, Hi*Wi]
        Output: B x (Ni, C)
        '''
        inst_feats = list()

        assert stu_feat_cur_level.size()[0] == len(inst_masks)
        B = len(inst_masks)

        for batch_idx in range(B):
            # (Ni, C)
            pool_feat= torch.mm(inst_masks[batch_idx], stu_feat_cur_level[batch_idx].T)
            # (Ni,)
            normalizer = inst_masks[batch_idx].sum(dim=-1)
            normalizer = torch.maximum(normalizer, torch.ones_like(normalizer))
            # object size-invariant by normalize the taken pixels
            pool_feat = pool_feat / normalizer[:, None]
            inst_feats.append(pool_feat)
        # B x (Ni, C)
        return inst_feats

    # Decouple Rendering (local instances and the global context)
    def rendering(self, batch_1DTo2D_attn_outputs, batchified_inside_masks, nr_gt_boxes, hws, B):
        '''
        Input:
            batch_1DTo2D_attn_outputs: F x (T, C)
            batchified_inside_masks: list of inside mask: F x B x (Ni, HiWi)

        process:
        (T, C) splitted into (Ni, C), i=1,...,B
        For each image: (C, Ni) x (Ni, HiWi) -> (C, HiWi), stack them to obtain (B, C, HiWi) -> (B, C, Hi, Wi)
        '''

        if self.add_context_box == True and getattr(self, 'render_using_bg_only', False) == False:
            # F x B x (Ni, C)
            batchified_attn_outputs = [batch_1DTo2D_attn_output_per_level.split(nr_gt_boxes, dim=0) for batch_1DTo2D_attn_output_per_level in batch_1DTo2D_attn_outputs]

            # split the attn_outputs
            # F x B x (Ni-1, C)
            batchified_inst_attn_outputs = [[self.local_inst_proj_1D(feats_per_level_per_img[:-1, :]) \
                    for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_attn_outputs]

            # F x B x (C, )
            batchified_ctx_attn_outputs = [[feats_per_level_per_img[-1] for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_attn_outputs]

            # split the masks
            # F x B x (Ni-1, HiWi)
            batchified_inst_inside_masks = [[feats_per_level_per_img[:-1, :] for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_inside_masks]
            # F x B x (HiWi, )
            #batchified_ctx_inside_masks = [[feats_per_level_per_img[-1] for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_inside_masks]

            # (1) get local instance feature map
            # F x B x (C, HiWi)
            batchified_attn_outputs_warped = [[torch.mm(attn_output_per_level_cur_img.T, inside_mask_per_level_cur_img) \
                    for attn_output_per_level_cur_img, inside_mask_per_level_cur_img in zip(batchified_attn_outputs_per_level, batchified_inside_masks_per_level)] \
                    for batchified_attn_outputs_per_level, batchified_inside_masks_per_level in zip(batchified_inst_attn_outputs, batchified_inst_inside_masks)]

            # F x (B, C, Hi, Wi)
            raw_interact_tea_feats = [torch.cat(batchified_attn_outputs_per_level, dim=0).reshape(B, -1, *hw) \
                    for batchified_attn_outputs_per_level, hw in zip(batchified_attn_outputs_warped, hws)]

            inst_featmaps = [self.local_inst_proj_2D(feat) for feat in raw_interact_tea_feats]

            # (2) get global context feature vector
            # F x (B, C)
            ctx_features = [self.global_ctx_proj_1D(torch.stack(feats_per_level, dim=0)) for feats_per_level in batchified_ctx_attn_outputs]

            raw_interact_tea_feats = [F.relu(inst_featmap + ctx_feature[:, :, None, None]) for inst_featmap, ctx_feature in zip(inst_featmaps, ctx_features)]

        elif self.add_context_box == False and getattr(self, 'render_using_bg_only', False) == False:
            # F x B x (Ni, C)
            batchified_attn_outputs = [batch_1DTo2D_attn_output_per_level.split(nr_gt_boxes, dim=0) for batch_1DTo2D_attn_output_per_level in batch_1DTo2D_attn_outputs]

            # split the attn_outputs
            # F x B x (Ni-1, C)
            batchified_inst_attn_outputs = [[self.local_inst_proj_1D(feats_per_level_per_img) \
                    for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_attn_outputs]

            # F x B x (C, )
            #batchified_ctx_attn_outputs = [[feats_per_level_per_img[-1] for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_attn_outputs]

            # split the masks
            # F x B x (Ni, HiWi)
            batchified_inst_inside_masks = [[feats_per_level_per_img for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_inside_masks]
            # F x B x (HiWi, )
            #batchified_ctx_inside_masks = [[feats_per_level_per_img[-1] for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_inside_masks]

            # (1) get local instance feature map
            # F x B x (C, HiWi)
            batchified_attn_outputs_warped = [[torch.mm(attn_output_per_level_cur_img.T, inside_mask_per_level_cur_img) \
                    for attn_output_per_level_cur_img, inside_mask_per_level_cur_img in zip(batchified_attn_outputs_per_level, batchified_inside_masks_per_level)] \
                    for batchified_attn_outputs_per_level, batchified_inside_masks_per_level in zip(batchified_inst_attn_outputs, batchified_inst_inside_masks)]

            # F x (B, C, Hi, Wi)
            raw_interact_tea_feats = [torch.cat(batchified_attn_outputs_per_level, dim=0).reshape(B, -1, *hw) \
                    for batchified_attn_outputs_per_level, hw in zip(batchified_attn_outputs_warped, hws)]

            inst_featmaps = [self.local_inst_proj_2D(feat) for feat in raw_interact_tea_feats]

            # (2) get global context feature vector
            # F x (B, C)
            #ctx_features = [self.global_ctx_proj_1D(torch.stack(feats_per_level, dim=0)) for feats_per_level in batchified_ctx_attn_outputs]

            # Additive Fusion
            #NOTE: is the relu here would harm the convergence ?

            raw_interact_tea_feats = [F.relu(inst_featmap) for inst_featmap in inst_featmaps]

        elif getattr(self, 'render_using_bg_only', False) == True:
            assert self.add_context_box == True
            # F x B x (Ni, C)
            batchified_attn_outputs = [batch_1DTo2D_attn_output_per_level.split(nr_gt_boxes, dim=0) for batch_1DTo2D_attn_output_per_level in batch_1DTo2D_attn_outputs]
            # fetch only the bg part
            # F x B x (C, )
            batchified_ctx_attn_outputs = [[feats_per_level_per_img[-1] for feats_per_level_per_img in feats_per_level] for feats_per_level in batchified_attn_outputs]

            # get global context feature vector
            # F x (B, C) -> F x (B, C, Hi, Wi)
            ctx_features = [self.global_ctx_proj_1D(torch.stack(feats_per_level, dim=0)) for feats_per_level in batchified_ctx_attn_outputs]

            raw_interact_tea_feats = [F.relu(ctx_feature).unsqueeze(-1).unsqueeze(-1).expand(B, self.nr_fpn_channels, *hw) for ctx_feature, hw in zip(ctx_features, hws)]

        return raw_interact_tea_feats


    def interactive_remapping(self, ori_canoni_embed, boxlists, ori_stu_fpn_feats, img_size_dict, batched_inputs):
        '''
        Do both (1) inter-object relation adaption and (2) intra-object knowledge mapping.
        Inputs:
            ori_canoni_embed, dubbed `label embeddings`: (T, C) where T = N1+...+NB, denoting all instance number appearing in annotations of current minibatch of images.
            boxlists: nested list of B x (Ni, 4), n might vary from different image. i = 1, ..., B
            ori_stu_fpn_feats that we are going to extract `appearance embeddings` from it by mask pooling: Dict of (B, C, Hi, Wi), 1 <= i <= 5, keys should be 'p3', 'p4', 'p5', 'p6', 'p7'
            img_size_dict:   {'h', h, 'w', w}, denoting the input image size of current minibatch of images
        Output:
            interact_tea_feats: Dict of (B, C, Hi, Wi), identical shape to `ori_stu_fpn_feat`
        '''
        if self.detach_appearance_embed == True:
            ori_stu_fpn_feats = {key: ori_stu_fpn_feats[key].detach() for key in ori_stu_fpn_feats.keys()}
        # list of B
        nr_gt_boxes = [len(boxlist) for boxlist in boxlists]
        B = len(nr_gt_boxes)
        hws = [ori_stu_fpn_feats[key].size()[-2:] for key in ori_stu_fpn_feats.keys()]
        L = len(hws)

        # (T, C)
        canoni_embed = self.canoni_proj_1D(ori_canoni_embed)
        # (T, 1, C)
        canoni_embed_unsqueeze = canoni_embed.unsqueeze(1)
        device = canoni_embed_unsqueeze.device

        # F x [B, C, Hi, Wi]
        stu_fpn_feats = {key: self.student_proj_2D(ori_stu_fpn_feats[key]) for key in ori_stu_fpn_feats.keys()}

        # list of inside mask: F x B x (Ni, HiWi)
        if self.use_seg_map:
            batchified_inside_masks = get_segmask_inside_gt(hws, batched_inputs, resolution(img_size_dict['h'], img_size_dict['w']), device, self.add_context_box)
        else:
            batchified_inside_masks = [[get_inside_gt_mask(boxlist, resolution(img_size_dict['h'], img_size_dict['w']), resolution(*hw), device) \
                    for boxlist in boxlists] for hw in hws]

        # Extract sparse instance features via instance masks from student feature maps
        # F x (T, 1, C)
        stu_embed_all_levels = list()

        # mask pooling
        for key, mask_per_level in zip(stu_fpn_feats.keys(), batchified_inside_masks):
            # B x (Ni, C)
            stu_embed_repeat = self.aggregate_per_level(stu_fpn_feats[key].flatten(start_dim=2), mask_per_level)
            # (T, 1, C)
            stu_embed_all_levels.append(torch.cat(stu_embed_repeat, dim=0).unsqueeze(1))

        box_to_batchLabel = [[i] * nr_gt_boxes[i] for i in range(B)]
        # (T, 1)
        box_to_batchLabel = torch.LongTensor(reduce(list.__add__, box_to_batchLabel)).reshape(-1, 1)
        # (T, T) mask, True for ignored elements
        attn_mask = (box_to_batchLabel != box_to_batchLabel.T).to(device)

        if self.interact_pattern == 'student_fill':
            batch_1DTo2D_attn_outputs = [query.squeeze(1) for query in stu_embed_all_levels]
        elif self.interact_pattern == 'teacher_fill':
            batch_1DTo2D_attn_outputs = [canoni_embed_unsqueeze.squeeze(1) for _ in stu_embed_all_levels]
        elif self.interact_pattern == 'stuGuided':
            # Query: (T, 1, C); label embeddings
            # Key and Value: (T, 1, C); student fpn instanced masked embeddings
            # attn_output: (T, 1, C)
            # q, k, v are all (T, 1, C), we get L x (T, C)
            batch_1DTo2D_attn_outputs = [self.multi_head_attn(query, canoni_embed_unsqueeze, canoni_embed_unsqueeze, attn_mask=attn_mask)[0].squeeze(1) for query in stu_embed_all_levels]

        elif self.interact_pattern == 'labelGuided':
            batch_1DTo2D_attn_outputs = [self.multi_head_attn(canoni_embed_unsqueeze, kv, kv, attn_mask=attn_mask)[0].squeeze(1) for kv in stu_embed_all_levels]
        else:
            raise ValueError('interact pattern: {} not supported !'.format(self.interact_pattern))

        # F x [B, C, Hi, Wi]
        raw_interact_tea_feats = self.rendering(batch_1DTo2D_attn_outputs, batchified_inside_masks, nr_gt_boxes, hws, B)

        interact_tea_feats = {key : self.refinement_module(raw_interact_tea_feats[i]) \
                for i, key in enumerate(stu_fpn_feats.keys())}

        return interact_tea_feats, batchified_inside_masks

    def forward(self, info_list):
        '''
        Input:
            info_list: input information: a tuple of (batched_inputs, images, r_features, features)
        Output:
            interact_tea_feats: list of (B, C, Hi, Wi), same shape as stu_fpn_feat
            trans_matrice_desc: (T*R, k, k)
            trans_matrice_feat: (T*R, 64, 64)
            inst_labels: B x (Ni,) if not add_context_box, else B x (Ni-1,), considering only the real instance existent in images
            batchified_inside_masks: F x B x (Ni, HiWi)
        '''
        # Extract Label Embeddings
        x, _, _, boxlists, img_size_dict, inst_labels = self.label_encoder_(info_list)
        # Finally Obtain mapped `instructive knowledge`, dubbed interact_tea_feats here.
        interact_tea_feats, batchified_inside_masks = self.interactive_remapping(x, boxlists, info_list[-1], img_size_dict, info_list[0])

        return interact_tea_feats, inst_labels, batchified_inside_masks
