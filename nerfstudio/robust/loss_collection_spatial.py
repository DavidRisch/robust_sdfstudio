import torch
from typing import Dict, List, Tuple, Optional
from torchtyping import TensorType

from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_base import LossCollectionBase


class LossCollectionSpatialBase(LossCollectionBase):
    """
    A LossCollection with the same shape as an image or an image path.
    The first two dimensions are height and width.
    May or may not contain missing values.
    """

    def __init__(self):
        super().__init__()
