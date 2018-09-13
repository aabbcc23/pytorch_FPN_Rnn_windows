# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from lib.model.utils.config import cfg
from lib.model.nms.nms_gpu import nms_gpu
from lib.model.nms.nms_cpu import nms_cpu
def nms(dets, thresh, force=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    # ---numpy version---
    if force==False:
        return nms_cpu(dets, thresh)
    else:
    # ---pytorch version---
        return nms_gpu(dets, thresh)
