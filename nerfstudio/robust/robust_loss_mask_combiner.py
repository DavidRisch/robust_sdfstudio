import torch
from typing import TYPE_CHECKING

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial

if TYPE_CHECKING:
    from nerfstudio.models.base_surface_model import SurfaceModelConfig


class RobustLossMaskCombiner:
    """
    Combines the masks for rgb, depth and normal.
    """

    @classmethod
    def combine(cls, loss_collection: LossCollectionDenseSpatial, count_required_for_distracted: int,
                device: torch.device):
        sum_of_masks = torch.zeros_like(loss_collection.rgb_mask, device=device)

        # print_tensor("before combine rgb", loss_collection.rgb_mask)

        sum_of_masks += loss_collection.rgb_mask
        sum_of_masks += loss_collection.depth_mask
        sum_of_masks += loss_collection.normal_mask

        # print_tensor("sum_of_masks", sum_of_masks)

        new_mask = (sum_of_masks >= count_required_for_distracted).float()

        # print_tensor("new_mask", new_mask)

        loss_collection.rgb_mask = new_mask
        loss_collection.depth_mask = new_mask
        loss_collection.normal_mask = new_mask

    @classmethod
    @torch.no_grad()
    def maybe_combine_masks(cls, loss_collection: LossCollectionDenseSpatial,
                            config: "SurfaceModelConfig", device: torch.device) -> None:

        if config.robust_loss_combine_mode == "Off":
            return
        elif config.robust_loss_combine_mode == "AnyDistracted":  # a pixel if distracted, if *any* of the components are distracted
            cls.combine(loss_collection=loss_collection, count_required_for_distracted=1, device=device)
        elif config.robust_loss_combine_mode == "AllDistracted":  # a pixel if distracted, if *every* single component is distracted
            cls.combine(loss_collection=loss_collection, count_required_for_distracted=3, device=device)
        elif config.robust_loss_combine_mode == "Majority":  # a pixel if distracted, if *most* of the components are distracted
            cls.combine(loss_collection=loss_collection, count_required_for_distracted=2, device=device)
        else:
            raise RuntimeError("Unknown value for robust_loss_combine_mode: " + config.robust_loss_combine_mode)
