import torch
from typing import Dict, List, Tuple, Optional, Callable
from torchtyping import TensorType


from nerfstudio.robust.loss_collection_sparse_spatial import LossCollectionSparseSpatial
from nerfstudio.robust.loss_collection_spatial import LossCollectionSpatialBase
from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_base import LossCollectionBase


class LossCollectionDenseSpatial(LossCollectionSpatialBase):
    """
    A LossCollection with the same shape as an image or a continuous image patch.
    The first two dimensions are height and width.
    """

    def __init__(self, offset_x: int, offset_y: int):
        super().__init__()

        self.offset_x: int = offset_x
        self.offset_y: int = offset_y

    def apply_function_to_masks(self, function: Callable[[torch.Tensor], torch.Tensor]):
        relevant_tensor_attribute_names = ["rgb_mask", "depth_mask", "normal_mask"]

        for attribute_name in relevant_tensor_attribute_names:
            old_value = getattr(self, attribute_name)

            modified_value = function(old_value)

            setattr(self, attribute_name, modified_value)

    def _make_attribute_spatial(self, dense_spatial: TensorType[...],
                                image_width: int, image_height: int,
                                device: torch.device):
        # print_tensor("make_attribute_spatial dense_spatial", dense_spatial)
        if dense_spatial.dtype == torch.float32:
            fill_value = torch.nan
        elif dense_spatial.dtype == torch.long:
            fill_value = -1
        else:
            assert False, dense_spatial.dtype
        sparse_spatial = torch.full((image_height, image_width), fill_value=fill_value, dtype=dense_spatial.dtype,
                                    device=device)
        # print_tensor("make_attribute_spatial sparse_spatial", sparse_spatial)
        sparse_spatial[self.pixel_coordinates_y, self.pixel_coordinates_x] = dense_spatial

        return sparse_spatial

    def make_into_sparse_spatial(self, image_width: int, image_height: int,
                                 device: torch.device) -> LossCollectionSparseSpatial:
        spatial_loss_collection = LossCollectionSparseSpatial()

        assert self.pixelwise_rgb_loss is not None

        for attribute_name in self.tensor_attribute_names:
            unordered_value = getattr(self, attribute_name)
            # print_tensor(f"make_into_spatial {attribute_name} unordered_value", unordered_value)
            spatial_value = self._make_attribute_spatial(
                dense_spatial=unordered_value,
                image_width=image_width,
                image_height=image_height,
                device=device
            )
            # print_tensor(f"make_into_spatial {attribute_name} ordered_value", spatial_value)
            setattr(spatial_loss_collection, attribute_name, spatial_value)

        spatial_loss_collection.valid_depth_pixel_count = self.valid_depth_pixel_count

        return spatial_loss_collection

    def make_into_unordered(self) -> "LossCollectionUnordered":
        from nerfstudio.robust.loss_collection_unordered import LossCollectionUnordered
        unordered_loss_collection = LossCollectionUnordered()

        assert self.pixelwise_rgb_loss is not None

        for attribute_name in self.tensor_attribute_names:
            unordered_value = getattr(self, attribute_name)
            # print_tensor(f"make_into_spatial {attribute_name} unordered_value", unordered_value)
            if unordered_value is not None:
                spatial_value = torch.flatten(unordered_value)
                # print_tensor(f"make_into_spatial {attribute_name} ordered_value", spatial_value)
                setattr(unordered_loss_collection, attribute_name, spatial_value)

        unordered_loss_collection.valid_depth_pixel_count = self.valid_depth_pixel_count

        return unordered_loss_collection
