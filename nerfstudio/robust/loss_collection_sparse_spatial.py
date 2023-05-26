import torch
from typing import Dict, List, Tuple, Optional
from torchtyping import TensorType

from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_spatial import LossCollectionSpatialBase


class LossCollectionSparseSpatial(LossCollectionSpatialBase):
    """
    A LossCollection with the same shape as an image or a continuous image patch.
    The first two dimensions are height and width.
    NaN (for float images) and -1 (for int images) mean that no data is available for that pixel.
    """

    def __init__(self):
        super().__init__()
