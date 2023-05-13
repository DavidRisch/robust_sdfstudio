#!/usr/bin/env python


from __future__ import annotations

import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from rich.console import Console

from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import Trainer
from nerfstudio.utils import comms, profiler

from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    MultiViewLoss,
    ScaleAndShiftInvariantLoss,
    SensorDepthLoss,
    compute_scale_and_shift,
    monosdf_normal_loss,
)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


def main():
    # check that ScaleAndShiftInvariantLoss works

    for gt in [
        torch.tensor([1, 2, 3, 4, 5]).float(),
        torch.tensor([10, 20, 30, 40, 50]).float()
    ]:
        print("\n==================\ngt", gt)
        mask = torch.tensor([True, True, True, True, True])

        for pred_1 in [gt, torch.tensor([1, 2, 3, 4, gt[4] * 2]).float()]:
            print("\npred_1:", pred_1)

            print("\n Test with pred_1")
            pred = pred_1
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            print("\n Test with pred_1 + ?")
            pred = pred_1 + 0.1
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            pred = pred_1 + 0.2
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            pred = pred_1 + 1
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            print("\n Test with pred_1 * ?")
            pred = pred_1 * 1.1
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            pred = pred_1 * 1.2
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            pred = pred_1 * 2
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            print("\n Test with pred_1 * ? + ?")
            pred = pred_1 * 1.1 + 1
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            pred = pred_1 * 1.2 + 10
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)

            pred = pred_1 * 2 + 5
            print("pred", pred)
            loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1).forward(prediction=pred.reshape(1, 1, -1),
                                                                           target=gt.reshape(1, 1, -1),
                                                                           mask=mask.reshape(1, 1, -1))
            print("loss", loss)


if __name__ == "__main__":
    main()
