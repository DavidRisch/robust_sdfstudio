import torch
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
from torchtyping import TensorType
import torch.nn.functional as F

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial
from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_unordered import LossCollectionUnordered

if TYPE_CHECKING:
    from nerfstudio.models.base_surface_model import SurfaceModelConfig


class RobustLossMaskCreator:
    """
    Create masks based on pixelwise losses.
    """

    def _create_loss_mask_from_loss(self, loss: TensorType, percentile: float) -> torch.Tensor:
        assert 0 <= percentile <= 100
        # print_tensor("loss", loss)
        sorted_loss_values, sorted_loss_indices = torch.sort(loss)
        # print_tensor("sorted_loss_values", sorted_loss_values)
        # print_tensor("sorted_loss_indices", sorted_loss_indices)

        total_length = len(loss)
        # print("total_length", total_length)
        keep_length = int(float(total_length) * percentile / 100.0)
        # print("keep_length", keep_length)

        # sorted_loss_values_keep = sorted_loss_values[:keep_length]
        sorted_loss_indices_keep = sorted_loss_indices[:keep_length]

        # print_tensor("sorted_loss_values_keep", sorted_loss_values_keep)
        # print_tensor("sorted_loss_indices_keep", sorted_loss_indices_keep)

        mask = torch.zeros(loss.shape, dtype=torch.float, device=loss.get_device())

        mask[sorted_loss_indices_keep] = 1

        return mask

    def maybe_create_loss_masks_from_losses(self, loss_collection: LossCollectionUnordered,
                                            config: "SurfaceModelConfig") -> None:

        # print_tensor("apply rgb_distracted_mask before", loss_collection.rgb_mask)
        if config.rgb_mask_from_percentile_of_rgb_loss != -1.0:
            assert not config.use_rgb_distracted_mask_for_rgb_loss_mask
            loss_collection.rgb_mask = self._create_loss_mask_from_loss(
                loss=loss_collection.pixelwise_rgb_loss,
                percentile=config.rgb_mask_from_percentile_of_rgb_loss,
            )
        if config.normal_mask_from_percentile_of_normal_loss != -1.0:
            assert not config.use_rgb_distracted_mask_for_normal_loss_mask
            loss_collection.normal_mask = self._create_loss_mask_from_loss(
                loss=loss_collection.get_pixelwise_normal_loss(),
                percentile=config.normal_mask_from_percentile_of_normal_loss,
            )
        if config.depth_mask_from_percentile_of_depth_loss != -1.0:
            assert not config.use_rgb_distracted_mask_for_depth_loss_mask
            loss_collection.depth_mask = self._create_loss_mask_from_loss(
                loss=loss_collection.pixelwise_depth_loss,
                percentile=config.depth_mask_from_percentile_of_depth_loss,
            )
