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

        combined_loss_collection.pixel_coordinates_x = torch.cat(
            [loss_collection.pixel_coordinates_x for loss_collection in loss_collections], dim=0)
        combined_loss_collection.pixel_coordinates_y = torch.cat(
            [loss_collection.pixel_coordinates_y for loss_collection in loss_collections], dim=0)

        combined_loss_collection.loss_collection_id = torch.cat(
            [loss_collection.loss_collection_id for loss_collection in loss_collections], dim=0)

        combined_loss_collection.pixelwise_rgb_loss = torch.cat(
            [loss_collection.pixelwise_rgb_loss for loss_collection in loss_collections], dim=0)

        combined_loss_collection.pixelwise_depth_loss = torch.cat(
            [loss_collection.pixelwise_depth_loss for loss_collection in loss_collections], dim=0)
        combined_loss_collection.valid_depth_pixel_count = sum(
            [loss_collection.valid_depth_pixel_count for loss_collection in loss_collections])

        combined_loss_collection.pixelwise_normal_l1 = torch.cat(
            [loss_collection.pixelwise_normal_l1 for loss_collection in loss_collections], dim=0)
        combined_loss_collection.pixelwise_normal_cos = torch.cat(
            [loss_collection.pixelwise_normal_cos for loss_collection in loss_collections], dim=0)

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

        spatial_loss_collection.pixel_coordinates_x = self.make_attribute_spatial(self.pixel_coordinates_x, image_width,
                                                                                  image_height)
        spatial_loss_collection.pixel_coordinates_y = self.make_attribute_spatial(self.pixel_coordinates_y, image_width,
                                                                                  image_height)
        spatial_loss_collection.loss_collection_id = self.make_attribute_spatial(self.loss_collection_id, image_width,
                                                                                 image_height)

        spatial_loss_collection.pixelwise_rgb_loss = self.make_attribute_spatial(self.pixelwise_rgb_loss, image_width,
                                                                                 image_height)

        spatial_loss_collection.pixelwise_depth_loss = self.make_attribute_spatial(self.pixelwise_depth_loss,
                                                                                   image_width,
                                                                                   image_height)

        spatial_loss_collection.pixelwise_normal_l1 = self.make_attribute_spatial(self.pixelwise_normal_l1, image_width,
                                                                                  image_height)
        spatial_loss_collection.pixelwise_normal_cos = self.make_attribute_spatial(self.pixelwise_normal_cos,
                                                                                   image_width,
                                                                                   image_height)

        return spatial_loss_collection
