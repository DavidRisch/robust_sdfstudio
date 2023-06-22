import torch
from typing import TYPE_CHECKING

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial
from nerfstudio.robust.print_utils import print_tensor

if TYPE_CHECKING:
    from nerfstudio.models.base_surface_model import SurfaceModelConfig


class RobustLossMaskCombiner:
    """
    Combines the masks for rgb, depth and normal.
    """

    @classmethod
    def combine(cls, loss_collection: LossCollectionDenseSpatial, distracted_votes_required_for_distracted_result: int,
                device: torch.device):
        clean_votes = torch.zeros_like(loss_collection.rgb_mask, device=device)

        # print_tensor("before combine rgb", loss_collection.rgb_mask)

        clean_votes += loss_collection.rgb_mask
        clean_votes += loss_collection.depth_mask
        clean_votes += loss_collection.normal_mask

        # print_tensor("clean_votes", clean_votes)

        distracted_votes = 3 - clean_votes
        # print_tensor("distracted_votes", distracted_votes)

        new_distracted_mask = (distracted_votes >= distracted_votes_required_for_distracted_result).float()
        new_clean_mask = torch.logical_not(new_distracted_mask).float()

        # print_tensor("new_clean_mask", new_clean_mask)

        loss_collection.rgb_mask = new_clean_mask
        loss_collection.depth_mask = new_clean_mask
        loss_collection.normal_mask = new_clean_mask

    @classmethod
    @torch.no_grad()
    def maybe_combine_masks(cls, loss_collection: LossCollectionDenseSpatial,
                            config: "SurfaceModelConfig", device: torch.device) -> None:

        if config.robust_loss_combine_mode == "Off":
            return
        elif config.robust_loss_combine_mode == "AnyDistracted":  # a pixel if distracted, if *any* of the components are distracted
            cls.combine(loss_collection=loss_collection, distracted_votes_required_for_distracted_result=1,
                        device=device)
        elif config.robust_loss_combine_mode == "AllDistracted":  # a pixel if distracted, if *every* single component is distracted
            cls.combine(loss_collection=loss_collection, distracted_votes_required_for_distracted_result=3,
                        device=device)
        elif config.robust_loss_combine_mode == "Majority":  # a pixel if distracted, if *most* of the components are distracted
            cls.combine(loss_collection=loss_collection, distracted_votes_required_for_distracted_result=2,
                        device=device)
        else:
            raise RuntimeError("Unknown value for robust_loss_combine_mode: " + config.robust_loss_combine_mode)
