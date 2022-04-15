# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import log_first_n

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler

from detectron2.evaluation.evaluator import *
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.solver import WarmupMultiStepLR, WarmupCosineLR

from detectron2.config import CfgNode

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_batch_data_loader",
    "build_detection_train_loader",
    "build_detection_test_loader",
    "get_detection_dataset_dicts",
    "load_proposals_into_dataset",
    "print_instances_class_histogram",
]


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in annotations
            if "keypoints" in ann
        )

    dataset_dicts = [
        x for x in dataset_dicts if visible_keypoints_in_image(x) >= min_keypoints_per_image
    ]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with fewer than {} keypoints.".format(
            num_before - num_after, min_keypoints_per_image
        )
    )
    return dataset_dicts


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)

    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}

    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (list[str]): a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if has_instances:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            check_metadata_consistency("thing_classes", dataset_names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def build_batch_data_loader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0
):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True
        )  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )


def build_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be ``DatasetMapper(cfg, True)``.

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_detection_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def my_inference_on_dataset(model, data_loader, evaluator, eval_teacher=False):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, eval_teacher=eval_teacher)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


# distinct optimizer for student and teacher (Under conventional paradigm, this is not necessary,
# set mainly for labelGen (with dynamic student-dependent teacher)
def build_distillator_optimizer(cfg, network):
    solver_stu, solver_tea = cfg.MODEL.DISTILLATOR.STUDENT.SOLVER, cfg.MODEL.DISTILLATOR.TEACHER.SOLVER
    def _get_params(model_list, base_lr, wd):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for model in model_list:
            for key, value in model.named_parameters(recurse=True):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = base_lr
                weight_decay = wd
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        return params

    # `student` and `adapter` use the same optimizer
    stu_params = _get_params([network.module.student, network.module.adapter], solver_stu.BASE_LR, solver_stu.WEIGHT_DECAY)
    tea_params = _get_params([network.module.teacher], solver_tea.BASE_LR, solver_tea.WEIGHT_DECAY)

    def _get_optim(optimizer_type, params, base_lr, **kwargs):
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(params, base_lr, momentum=kwargs['momentum'])
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(params, base_lr, betas=(0.9, 0.999))
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    # use cfg.MODEL.DISTILLATOR.{STUDENT/TEACHER}_xx rather than cfg.SOLVER.xx
    stu_optim = _get_optim(solver_stu.OPTIMIZER, stu_params, solver_stu.BASE_LR, momentum=solver_stu.MOMENTUM)
    tea_optim = _get_optim(solver_tea.OPTIMIZER, tea_params, solver_tea.BASE_LR, momentum=solver_tea.MOMENTUM)

    return stu_optim, tea_optim

def build_distillator_lr_scheduler(
        solver: CfgNode, optimizer: torch.optim.Optimizer
        ) -> torch.optim.lr_scheduler._LRScheduler:
    name = solver.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
                optimizer,
                solver.STEPS,
                solver.GAMMA,
                warmup_factor=solver.WARMUP_FACTOR,
                warmup_iters = solver.WARMUP_ITERS,
                warmup_method = solver.WARMUP_METHOD,
                )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
                optimizer,
                solver.MAX_ITER,
                warmup_factor=solver.WARMUP_FACTOR,
                warmup_iters = solver.WARMUP_ITERS,
                warmup_method = solver.WARMUP_METHOD,
                )
    else:
        raise ValueError("Unknown LR sheduler: {}".format(name))

from detectron2.config import CfgNode as CN

def build_distillator_configs(cfg):
    cfg.NUM_CLASSES = 80
    cfg.MODEL.DISTILLATOR = CN()
    cfg.MODEL.DISTILLATOR.STUDENT = CN()
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER = CN()
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.OPTIMIZER = 'SGD'
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.BASE_LR = 0.02
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.MOMENTUM = 0.9
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.LR_SCHEDULER_NAME = None
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.STEPS = None
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.GAMMA = None
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.WARMUP_FACTOR = None
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.WARMUP_ITERS = None
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.WARMUP_METHOD = None

    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.AMP = CN()
    cfg.MODEL.DISTILLATOR.STUDENT.SOLVER.AMP.ENABLED = False


    cfg.MODEL.DISTILLATOR.STUDENT.META_ARCH = None


    cfg.MODEL.DISTILLATOR.TEACHER = CN()
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER = CN()

    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.OPTIMIZER = 'SGD'
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.BASE_LR = 0.02
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.MOMENTUM = 0.9
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.LR_SCHEDULER_NAME = None
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.STEPS = None
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.GAMMA = None
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.WARMUP_FACTOR = None
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.WARMUP_ITERS = None
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.WARMUP_METHOD = None

    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.AMP = CN()
    cfg.MODEL.DISTILLATOR.TEACHER.SOLVER.AMP.ENABLED = False

    cfg.MODEL.DISTILLATOR.TEACHER.META_ARCH = None

    cfg.MODEL.DISTILLATOR.ADAPTER = CN()
    cfg.MODEL.DISTILLATOR.ADAPTER.META_ARCH = 'SequentialConvs'

    cfg.MODEL.DISTILLATOR.PRE_NONDISTILL_ITERS= 40000
    cfg.MODEL.DISTILLATOR.POST_NONDISTILL_ITERS = 0
    cfg.MODEL.DISTILLATOR.PRE_FREEZE_STUDENT_BACKBONE_ITERS = 10000

    cfg.MODEL.DISTILLATOR.DISTILL_OFF = 0
    cfg.MODEL.DISTILLATOR.DISTILL_ON = 1

    cfg.MODEL.RECIPROCAL_FPN_STRIDES = [1/8, 1/16, 1/32, 1/64, 1/128]


    #TODO: ?
    cfg.MODEL.LOAD_BOXMAP = False # add box_map key
    cfg.MODEL.STRONGER_AUGS = False # add extra_images to batched_input
    cfg.MODEL.LOAD_BOX_MASK = False


    cfg.MODEL.DISTILLATOR.HIDDEN_DIM = 64
    cfg.MODEL.DISTILLATOR.SMOOTH = 0

    cfg.MODEL.DISTILLATOR.EVAL_TEACHER = True

    #NOTE: Inter-object Relation Adapter and Intra-object Knowledge Mapper
    cfg.MODEL.DISTILLATOR.TEACHER.INTERACT_PATTERN = 'stuGuided'


    # box_format: 'x1y1x2y2' or 'x1y1wh'
    cfg.MODEL.DISTILLATOR.LABEL_ENCODER = CN()
    # data loading
    cfg.MODEL.DISTILLATOR.LABEL_ENCODER.LOAD_LABELMAP = False
    cfg.MODEL.DISTILLATOR.LABEL_ENCODER.BOX_FORMAT = 'x1y1x2y2'
    cfg.MODEL.DISTILLATOR.LABEL_ENCODER.CATEGORY_FORMAT = 'one_hot'

    cfg.MODEL.DISTILLATOR.TEACHER.NR_TRANSFORMER_HEADS = 8
    cfg.MODEL.DISTILLATOR.TEACHER.DETACH_APPEARANCE_EMBED = False

    cfg.MODEL.DISTILLATOR.TEACHER.ADD_CONTEXT_BOX = False

    cfg.MODEL.DISTILLATOR.KNOWLEDGE_MAPPER = CN()


    cfg.MODEL.DISTILLATOR.TEACHER.AFFINE = False

    cfg.MODEL.DISTILLATOR.LAMBDA = 1.0
    cfg.MODEL.DISTILLATOR.TOWER_DISTILL_COEF = 1.0
    cfg.MODEL.DISTILLATOR.USE_MTH_HEAD = 1
    cfg.MODEL.DISTILLATOR.DETACH_TEA_WHEN_DISTILL = True
    cfg.MODEL.DISTILLATOR.ADAIN_BEFORE_DISTILL = False

    cfg = build_fcos(cfg)
    cfg = build_swint(cfg)

    return cfg

def build_swint(cfg):
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False

    cfg.MODEL.FPN.TOP_LEVELS = 2

    return cfg


def build_fcos(cfg):
    cfg.MODEL.FCOS = CN(dict(
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            NUM_CONVS=4,
            FPN_STRIDES=[8, 16, 32, 64, 128],
            PRIOR_PROB=0.01,
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            NORM_SYNC=True,
            REG_WEIGHT=2.0, # for atss
    ))

    cfg.MODEL.SHIFT_GENERATOR = CN(dict(
            NUM_SHIFTS=1,
            OFFSET=0.5,
        ))
    cfg.MODEL.NMS_TYPE = 'normal'

    cfg.MODEL.POTO = CN(dict(
            ALPHA=0.8,
            CENTER_SAMPLING_RADIUS=1.5,
            REG_WEIGHT=2.0,
        ))
    cfg.MODEL.ATSS = CN(dict(
            ANCHOR_SCALE=8,
            TOPK=9,
        ))
    return cfg
