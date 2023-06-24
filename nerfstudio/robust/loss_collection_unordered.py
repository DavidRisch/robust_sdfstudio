import torch
from typing import Dict, List, Tuple, Optional, Union
from torchtyping import TensorType

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial
from nerfstudio.robust.loss_collection_sparse_spatial import LossCollectionSparseSpatial
from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_base import LossCollectionBase
from nerfstudio.robust.loss_collection_spatial import LossCollectionSpatialBase


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

    def _make_attribute_spatial(self, original: TensorType[...],
                                image_width: int, image_height: int,
                                offset_x: int, offset_y: int,
                                allow_sparse: bool, device: torch.device):
        # print_tensor("make_attribute_spatial original", original)
        if original.dtype == torch.float32:
            fill_value = torch.nan
        elif original.dtype == torch.long:
            fill_value = -1
        else:
            assert False, original.dtype
        spatial = torch.full((image_height, image_width), fill_value=fill_value, dtype=original.dtype, device=device)
        # print_tensor("make_attribute_spatial spatial", spatial)
        spatial[self.pixel_coordinates_y - offset_y, self.pixel_coordinates_x - offset_x] = original

        if not allow_sparse:
            if torch.any(spatial == fill_value):
                print_tensor("original", original)
                print_tensor("spatial", spatial)
                raise RuntimeError("Spatial value has missing values, but should not be sparse")

        return spatial

    def _make_into_spatial(self, spatial_loss_collection: LossCollectionSpatialBase,
                           image_width: int, image_height: int,
                           offset_x: int, offset_y: int,
                           allow_sparse: bool,
                           device: torch.device) -> None:
        assert self.pixelwise_rgb_loss is not None

        for attribute_name in self.tensor_attribute_names:
            unordered_value = getattr(self, attribute_name)
            if unordered_value is None:
                continue
            # print_tensor(f"make_into_spatial {attribute_name} unordered_value", unordered_value)
            spatial_value = self._make_attribute_spatial(
                original=unordered_value,
                image_width=image_width,
                image_height=image_height,
                offset_x=offset_x,
                offset_y=offset_y,
                allow_sparse=allow_sparse,
                device=device,
            )
            # print_tensor(f"make_into_spatial {attribute_name} ordered_value", spatial_value)
            setattr(spatial_loss_collection, attribute_name, spatial_value)

        spatial_loss_collection.valid_depth_pixel_count = self.valid_depth_pixel_count

    def make_into_dense_spatial(self, device: torch.device) -> LossCollectionDenseSpatial:
        x_min = torch.min(self.pixel_coordinates_x).item()
        x_max = torch.max(self.pixel_coordinates_x).item()
        y_min = torch.min(self.pixel_coordinates_y).item()
        y_max = torch.max(self.pixel_coordinates_y).item()

        image_width = x_max - x_min + 1
        image_height = y_max - y_min + 1
        offset_x = x_min
        offset_y = y_min

        # print(f"{image_width=}")
        # print(f"{image_height=}")

        spatial_loss_collection = LossCollectionDenseSpatial(
            offset_x=offset_x,
            offset_y=offset_y
        )

        self._make_into_spatial(
            spatial_loss_collection=spatial_loss_collection,
            image_width=image_width,
            image_height=image_height,
            offset_x=offset_x,
            offset_y=offset_y,
            allow_sparse=False,
            device=device,
        )

        return spatial_loss_collection

    def make_into_sparse_spatial(self, image_width: int, image_height: int,
                                 device: torch.device) -> LossCollectionSparseSpatial:
        spatial_loss_collection = LossCollectionSparseSpatial()

        self._make_into_spatial(
            spatial_loss_collection=spatial_loss_collection,
            image_width=image_width,
            image_height=image_height,
            offset_x=0,
            offset_y=0,
            allow_sparse=True,
            device=device,
        )

        return spatial_loss_collection
