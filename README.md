# LGD: Label-Guided Self-Distillation for Object Detection

This repository is an official implementation of the AAAI 2022 paper [LGD: Label-Guided Self-Distillation for Object Detection]
(https://arxiv.org/abs/2109.11496)

## Introduction
**TL; DR** We propose the first self-distillation framework for general object detection, termed LGD (Label-Guided self-Distillation).

**Abstract.** In this paper, we propose the first self-distillation framework for general object detection, termed LGD (**L**abel-**G**uided self-**D**istillation). Previous studies rely on a strong pretrained teacher to provide instructive knowledge that could be unavailable in real-world scenarios. Instead, we generate an instructive knowledge by inter-and-intra relation modeling among objects, requiring only student representations and regular labels. Concretely, our framework involves sparse label-appearance encoding, inter-object relation adaptation and intra-object knowledge mapping to obtain the instructive knowledge. They jointly form an implicit teacher at training phase, dynamically dependent on labels and evolving student representations. Modules in LGD are trained end-to-end with student detector and are discarded in inference. Experimentally, LGD obtains decent results on various detectors, datasets, and extensive tasks like instance segmentation. For example in MS-COCO dataset, LGD improves RetinaNet with ResNet-50 under 2x single-scale training from 36.2% to 39.0% mAP (+ **2.8**%). It boosts much stronger detectors like FCOS with ResNeXt-101 DCN v2 under 2x multi-scale training from 46.1% to 47.9% (+ **1.8**%).
Compared with a classical teacher-based method FGFI, LGD not only performs better without requiring pretrained teacher but also reduces **51**% training cost beyond inherent student learning.

## Main Results (About MS-COCO in regular and supplementary sections) 
Experiments are mainly conducted with 8x 2080 ti GPUs (For experiments like with backbone Swin-Tiny that are proned to be OOM, we opt for distributed training with 2 machines with total 16 2080 ti GPUs or single 8-GPUs V100 machine). Here we provide primary experimental results that detection heads with various backbones with FPN (Table 1, 12 and 13) in the paper.  We perform a code refactoring upon the primitive codes before paper submission deadline and re-run the experiments which could be well reproduced. Compared with the arXiv version of paper, there could be around 0.1 mAP difference (slightly +0.1 for the most part). lgd-pretrained models and logs are available below:

RetinaNet

Backbone | mAP | config | log | pretrained model
--- |:---:|:---:|:---:|:---:|
R-50 | 40.4 | [config](configs/Distillation/RetinaNet/retinanet_R_50_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1Tqk5n8tRMnvjSh2ezRNi24W0kX7u5t4o/view?usp=sharing) | [LINK](https://drive.google.com/file/d/1bZSCwrJpMSgmFS2W7D2cHrUqK-h2s1bd/view?usp=sharing) |
R-101 | 42.1 | [config](configs/Distillation/RetinaNet/retinanet_R_101_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1OeYaAg_AEcZTnuvvmYjhVgZ-Nlw_srvb/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/16m3EGALHVWFQljkgUd1_g2JrumO8xZMx/view?usp=sharing) |
R-101-DCN v2 | 44.5 | [config](configs/Distillation/RetinaNet/retinanet_R_101_dcnv2_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1mVLw3t2ERqjnbpAZXO9-sqU47jiSM1q3/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1j3zfzriHe09ki-D4UnSGEMQGfesCmkVY/view?usp=sharing) |
X-101-DCN v2 | 45.9 | [config](configs/Distillation/RetinaNet/retinanet_X_101_dcnv2_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistillIters=30k_postNonDistillIters=50k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1JRhyU658E-MWueH-O9hWdAvRdwemKA1p/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1vKxWEKM8Dmryaf4Q5m--A52_yS_Hi1o5/view?usp=sharing) |
Swin-Tiny | 45.9 | [config](configs/Distillation/RetinaNet/retinanet_Swin_Tiny_3xMS_stuGuided_addCtxBox\=YES_detachAppearanceEmbed\=NO_preNondistillIters\=30k_preFreezeStudentBackboneIters\=20k.yaml) | [LINK](https://drive.google.com/file/d/17W1jDrYvQsOrxeu39muuJ3oqOrWAETys/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1W0_YNP8POzZsbMtFfJhXv2CshZCyarth/view?usp=sharing) |

FCOS

Backbone | mAP | config | log | pretrained model
--- |:---:|:---:|:---:|:---:|
R-50 | 42.4 | [config](configs/Distillation/FCOS/fcos_R_50_2xMS_stuGuided_addCtxBox=NO_detachAppearanceEmbed=NO_preNondistilIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1QH1WYM1f-ahdli-E1Av3d2HbmCDAi7Cn/view?usp=sharing) | [LINK](https://drive.google.com/file/d/1JlRzUIXY7w1CvLCQ_mW0ZD6uvKaRUgGn/view?usp=sharing) |
R-101 | 44.0 | [config](configs/Distillation/FCOS/fcos_R_101_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistilIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1Or6OxT0rO-SDNAyjAYPr6ZHBtVuZtdYH/view?usp=sharing) | [LINK](https://drive.google.com/file/d/1uql08LtDbXRTGjIJZPMh40U74Bk99FdI/view?usp=sharing) |
R-101-DCN v2 | 46.3 | [config](configs/Distillation/FCOS/fcos_R_101_dcnv2_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistilIters=30k_postNondistillIters=50k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1AgYFTOCmipHB_Tu8pKZ26Jf2MuYOhgKs/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1Z06Kf92Jf1rJfCgIN7V9K2db4MUOEtvd/view?usp=sharing) |
X-101-DCN v2 | 47.9 | [config](configs/Distillation/FCOS/fcos_X_101_dcnv2_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistilIters=30k_postNondistillIters=50k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1KOsg72plN9AuiOIOCsH0F5lxdLR67Mbb/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1t21WmZ9FW_JFLoNnLpcIJQ1w9lBQZ32M/view?usp=sharing) |

Faster R-CNN

Backbone | mAP | config | log | pretrained model
--- |:---:|:---:|:---:|:---:|
R-50 | 40.5 | [config](configs/Distillation/FasterRCNN/faster_rcnn_R_50_2xMS_stuGuided_addCtxBox=NO_detachAppearanceEmbed=YES_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/19-E_q0BjClFqBSvGr8bmv9mIwbOT9NwD/view?usp=sharing) | [LINK](https://drive.google.com/file/d/1up1t1fsaJx3VMMXRN4gY1EpG3j1PVOb5/view?usp=sharing) |
R-101 | 42.2 | [config](configs/Distillation/FasterRCNN/faster_rcnn_R_101_2xMS_stuGuided_addCtxBox=NO_detachAppearanceEmbed=YES_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1rRcufB0vrk9vNAFun-969Vm3iaHHnWHH/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1sVvVUhLNjkr2ZsWGbwSn2EkCChaAtmPR/view?usp=sharing) |
R-101-DCN v2 | 44.8 | [config](configs/Distillation/FasterRCNN/faster_rcnn_R_101_dcnv2_2xMS_stuGuided_addCtxBox=NO_detachAppearanceEmbed=YES_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1JV7Pm9TqkoTmN3Y7gpG9niK8cm32V-lT/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1fNhQ96nGbzmKK6y34DB_GbvGpi_ok3dt/view?usp=sharing) |
X-101-DCN v2 | 46.2 | [config](configs/Distillation/FasterRCNN/faster_rcnn_X_101_dcnv2_2xMS_stuGuided_addCtxBox=NO_detachAppearanceEmbed=YES_preNondistillIters=30k_postNondistillIters=50k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1_LrO3EqHqxh1nxLxARBsYd-nDoWsYsXD/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1R7e4-7krtSbmz1Ogid4jeigASOQ5xts3/view?usp=sharing) |

Mask R-CNN

Backbone | AP(m) | AP(b) | config | log | pretrained model
--- |:---:|:---:|:---:|:---:|:---:|
Swin-Tiny | 42.5 | 46.4 | [config](configs/Distillation/MaskRCNN/mask_rcnn_Swin_Tiny_3xMS_stuGuided_addCtxBox=NO_detachAppearanceEmbed=YES_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml) | [LINK](https://drive.google.com/file/d/1GdVUiEurBGYCFJicMEnGbOFMQ6NVgV6m/view?usp=sharing) |
[LINK](https://drive.google.com/file/d/1OsA4r3lJUVBHnoZG_wN4gYOTHqkSulM_/view?usp=sharing) |




## Installation

This codebase is built upon [detectron2] (https://github.com/facebookresearch/detectron2)

### Requirements

* Ubuntu 16.04 LTS, CUDA>=10.0, GCC>=5.4.0
* Python>=3.6.12
* Virtual environment via Anaconda (>=4.10.3) is recommended:
   ```bash
   conda create -n lgd python=3.7 pip
   ```
   Activate it by
   ```bash
   conda activate lgd
   ```

* Pytorch>=1.7.1, torchvision>=0.8.2
* Other requirements
  ```
  pip install -r requirements.txt
  ```
* Get into the LGD code directory (denoted by ${PROJ}).
  ```bash
  cd ${PROJ}
  ```

## Usage
### Dataset preparation
For instance, downloading MS-COCO (https://cocodataset.org/) whose hierarchy is organized as follows:
MSCOCO  
      |_ annotations  
                    |_ instances_train2017.json  
	            |_ instances_val2017.json  
      |_ train2017  
      |_ val2017  

``` bash
mkdir ${PROJ}/datasets
ln -s /path/to/MSCOCO datasets/coco
```

### Training

```bash
python3 train.py --config-file ${CONFIG} --num-gpus ${NUM_GPUS} --resume
```
e.g. when it comes to RetinaNet R-101 2xMS on 8 gpu cards' setting. It looks like:
```bash
python3 train.py --config-file configs/Distillation/RetinaNet/retinanet_R_101_2xMS_stuGuided_addCtxBox=YES_detachAppearanceEmbed=NO_preNondistillIters=30k_preFreezeStudentBackboneIters=20k.yaml --num-gpus 8 --resume
```

### Evaluation
It is handy to add [--eval-only] option to turn training command into evaluation usage.
```bash
python3 train.py --eval-only --config-file ${CONFIG} MODEL.WEIGHTS ${SNAPSHOT} MODEL.DISTILLATOR.EVAL_TEACHER False
```

## Citing LGD
```bibtex
@article{zhang2021lgd,
  title={LGD: Label-guided Self-distillation for Object Detection},
  author={Zhang, Peizhen and Kang, Zijian and Yang, Tong and Zhang, Xiangyu and Zheng, Nanning and Sun, Jian},
  journal={arXiv preprint arXiv:2109.11496},
  year={2021}
}
```
