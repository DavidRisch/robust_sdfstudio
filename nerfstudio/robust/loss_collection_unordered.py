import torch
from typing import Dict, List, Tuple, Optional
from torchtyping import TensorType

from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_base import LossCollectionBase
from nerfstudio.robust.loss_collection_spatial import LossCollectionSpatial


class LossCollectionUnordered(LossCollectionBase):
    """
    LossCollection without a specific order to the pixels (e.g. a randomly selected batch of pixels).
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_combination(cls, loss_collections: List["LossCollectionUnordered"]) -> "LossCollectionUnordered":
        combined_loss_collection = cls()

        for index, loss_collection in enumerate(loss_collections):
            # print("....pixel_coordinates_x", loss_collection.pixel_coordinates_x)
            # print("....pixel_coordinates_y", loss_collection.pixel_coordinates_y)
            loss_collection.loss_collection_id = torch.ones_like(loss_collection.pixelwise_rgb_loss,
                                                                 dtype=torch.long) * index

        for attribute_name in combined_loss_collection.tensor_attribute_names:
            combined_value = torch.cat([
                getattr(loss_collection, attribute_name)
                for loss_collection in loss_collections
            ], dim=0)
            setattr(combined_loss_collection, attribute_name, combined_value)

        combined_loss_collection.valid_depth_pixel_count = sum(
            [loss_collection.valid_depth_pixel_count for loss_collection in loss_collections])

        return combined_loss_collection

    def make_attribute_spatial(self, original: TensorType[...], image_width: int, image_height: int):
        # print_tensor("make_attribute_spatial original", original)
        if original.dtype == torch.float32:
            fill_value = torch.nan
        elif original.dtype == torch.long:
            fill_value = -1
        else:
            assert False, original.dtype
        spatial = torch.full((image_height, image_width), fill_value=fill_value, dtype=original.dtype, device="cpu")
        # print_tensor("make_attribute_spatial spatial", spatial)
        spatial[self.pixel_coordinates_y, self.pixel_coordinates_x] = original
        # print_tensor("make_attribute_spatial spatial", spatial)
        return spatial

    def make_into_spatial(self, image_width: int, image_height: int) -> LossCollectionSpatial:
        assert self.pixelwise_rgb_loss is not None

        spatial_loss_collection = LossCollectionSpatial()

        for attribute_name in self.tensor_attribute_names:
            unordered_value = getattr(self, attribute_name)
            ordered_value = self.make_attribute_spatial(unordered_value, image_width, image_height)
            setattr(spatial_loss_collection, attribute_name, ordered_value)

        spatial_loss_collection.valid_depth_pixel_count = self.valid_depth_pixel_count

        return spatial_loss_collection
