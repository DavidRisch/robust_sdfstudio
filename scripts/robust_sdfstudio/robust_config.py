#!/usr/bin/env python


from __future__ import annotations

import os
from typing import Optional, List
import subprocess


class RobustConfig:
    """
    Contains configurations options for newly added robustness features.
    """

    def __init__(self,
                 use_gt_distracted_mask: bool = False,
                 robust_loss_kernel_name: str = "NoKernel",
                 simple_percentile: Optional[float] = None,
                 robust_loss_classify_patches_mode: str = "Off",
                 ):
        self.use_gt_distracted_mask = use_gt_distracted_mask
        self.robust_loss_kernel_name = robust_loss_kernel_name
        self.simple_percentile = simple_percentile
        self.robust_loss_kernel_name = robust_loss_kernel_name
        self.robust_loss_classify_patches_mode = robust_loss_classify_patches_mode
