# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------
#!/usr/bin/env python
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel

from detectron2.solver.build import maybe_add_gradient_clipping
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
)
from utils.build import (
    build_detection_test_loader,
    build_detection_train_loader,
    build_distillator_optimizer,
    build_distillator_lr_scheduler,
    my_inference_on_dataset,
    build_distillator_configs,
)
from detectron2.evaluation.evaluator import *
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

import models.distillator
import models.adapters

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    #NOTE: crowdHuman also use COCOEvaluator
    if evaluator_type in ["coco", "coco_panoptic_seg", "crowdHuman"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))

    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)


    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)



def do_test(cfg, model, eval_teacher=False):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = my_inference_on_dataset(model, data_loader, evaluator, eval_teacher=eval_teacher)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()

    stu_optimizer, tea_optimizer = build_distillator_optimizer(cfg, model)
    stu_scheduler = build_distillator_lr_scheduler(cfg.MODEL.DISTILLATOR.STUDENT.SOLVER, stu_optimizer)
    tea_scheduler = build_distillator_lr_scheduler(cfg.MODEL.DISTILLATOR.TEACHER.SOLVER, tea_optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, stu_optimizer=stu_optimizer, tea_optimizer=tea_optimizer,
        stu_scheduler=stu_scheduler, tea_scheduler=tea_scheduler)

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    # :class: `PeriodicCheckpointer` built in Detectron2 is a mainly a wrapper of the :class: `Checkpoint` :meth: save
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            if iteration < cfg.MODEL.DISTILLATOR.PRE_NONDISTILL_ITERS:
                model.module.distill_flag = cfg.MODEL.DISTILLATOR.DISTILL_OFF
            elif iteration > max_iter - cfg.MODEL.DISTILLATOR.POST_NONDISTILL_ITERS:
                model.module.distill_flag = cfg.MODEL.DISTILLATOR.DISTILL_OFF
            else:
                model.module.distill_flag = cfg.MODEL.DISTILLATOR.DISTILL_ON

            loss_dict = model(data)
            losses = sum(loss_dict.values())

            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            stu_optimizer.zero_grad()
            tea_optimizer.zero_grad()
            losses.backward()

            if iteration < cfg.MODEL.DISTILLATOR.PRE_FREEZE_STUDENT_BACKBONE_ITERS:
                for p in model.module.student.raw_backbone.parameters():
                    p.grad = None

            stu_optimizer.step()
            tea_optimizer.step()

            storage.put_scalar("stu_lr", stu_optimizer.param_groups[0]["lr"], smoothing_hint=False)
            storage.put_scalar("tea_lr", tea_optimizer.param_groups[0]["lr"], smoothing_hint=False)
            stu_scheduler.step()
            tea_scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                if cfg.MODEL.DISTILLATOR.EVAL_TEACHER:
                    logger.info('**************EVAL TEACHER***************')
                    do_test(cfg, model, True)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg = build_distillator_configs(cfg)
    cfg.merge_from_file(args.config_file)

    # Detectors currently supported.
    # primitive cfg.MODEL.META_ARCHITECTURE in ('RetinaNet', 'GeneralizedRCNN', 'FCOS', 'POTO', 'ATSS')
    if not 'Distillator' in cfg.MODEL.META_ARCHITECTURE:
        cfg.MODEL.META_ARCHITECTURE = 'Distillator' + cfg.MODEL.META_ARCHITECTURE

    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)

    model.to(torch.device(cfg.MODEL.DEVICE))
    logger.info("Model:\n{}".format(model))
    model.distill_flag = cfg.MODEL.DISTILLATOR.DISTILL_OFF
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.MODEL.DISTILLATOR.EVAL_TEACHER:
            logger.info('**************EVAL TEACHER***************')
            do_test(cfg, model, True)
            logger.info('**************EVAL TEACHER END***************')
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume)
    if cfg.MODEL.DISTILLATOR.EVAL_TEACHER:
        logger.info('**************EVAL TEACHER***************')
        do_test(cfg, model, True)
        logger.info('**************EVAL TEACHER END***************')
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # Helpful for distributed training with multiple machines if master node address is required.
    if args.machine_rank == 0:
        import subprocess
        master_ip = subprocess.check_output(['hostname', '--fqdn']).decode('utf-8')
        master_ip = str(master_ip).strip()
        args.dist_url = 'tcp://{}:23333'.format(master_ip)
        print(args.dist_url)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
