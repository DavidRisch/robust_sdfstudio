import torch
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
from torchtyping import TensorType
import torch.nn.functional as F

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial
from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_unordered import LossCollectionUnordered

if TYPE_CHECKING:
    from nerfstudio.models.base_surface_model import SurfaceModelConfig


class RobustLoss:
    """
    Central part of our robust loss project.
    """

    @staticmethod
    def maybe_get_loss_masks_from_distractor_mask(loss_collection: LossCollectionUnordered, batch: Dict,
                                                  config: "SurfaceModelConfig") -> None:
        if "rgb_distracted_mask" in batch:
            assert len(batch["rgb_distracted_mask"].shape) == 1
            rgb_distracted_mask = batch["rgb_distracted_mask"]

            # print_tensor("apply rgb_distracted_mask before", loss_collection.rgb_mask)
            if config.use_rgb_distracted_mask_for_rgb_loss_mask:
                assert config.rgb_mask_from_percentile_of_rgb_loss == -1.0
                loss_collection.rgb_mask[rgb_distracted_mask] = 0
                # print_tensor("apply rgb_distracted_mask after", loss_collection.rgb_mask)
            if config.use_rgb_distracted_mask_for_normal_loss_mask:
                assert config.rgb_mask_from_percentile_of_rgb_loss == -1.0
                loss_collection.depth_mask[rgb_distracted_mask] = 0
            if config.use_rgb_distracted_mask_for_depth_loss_mask:
                assert config.rgb_mask_from_percentile_of_rgb_loss == -1.0
                loss_collection.normal_mask[rgb_distracted_mask] = 0

    @staticmethod
    def _create_loss_mask_from_loss(loss: TensorType, percentile: float) -> torch.Tensor:
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

    @classmethod
    def maybe_apply_kernel_to_masks(cls, loss_collection: LossCollectionDenseSpatial,
                                    config: "SurfaceModelConfig", device: torch.device) -> None:

        if config.robust_loss_kernel_name == "NoKernel":
            kernel = None
        elif config.robust_loss_kernel_name == "Box_3x3":
            size = 3
            kernel = torch.ones((1, 1, size, size), dtype=torch.float32, device=device)
            kernel /= size ** 2
        elif config.robust_loss_kernel_name == "Box_5x5":
            size = 5
            kernel = torch.ones((1, 1, size, size), dtype=torch.float32, device=device)
            kernel /= size ** 2
        else:
            raise RuntimeError("Unknown value for robust_loss_kernel_name: " + config.robust_loss_kernel_name)

        # print_tensor("kernel", kernel)

        if kernel is not None:
            def apply_kernel(old_value):
                width, height = old_value.shape[1], old_value.shape[0]

                # print_tensor(f"{attribute_name} before convolution", old_value)
                old_value = old_value.reshape((1, 1, height, width))
                # print_tensor(f"{attribute_name} before convolution", old_value)

                modified_value = F.conv2d(input=old_value, weight=kernel, stride=1, padding="same")
                modified_value = (modified_value >= 0.5).float()

                # print_tensor(f"{attribute_name} after convolution", modified_value)
                modified_value = modified_value.reshape((height, width))
                # print_tensor(f"{attribute_name} after convolution", modified_value)

                return modified_value

            loss_collection.apply_function_to_masks(function=apply_kernel)

    @classmethod
    def maybe_create_loss_masks_from_losses(cls, loss_collection: LossCollectionUnordered,
                                            config: "SurfaceModelConfig") -> None:

        # print_tensor("apply rgb_distracted_mask before", loss_collection.rgb_mask)
        if config.rgb_mask_from_percentile_of_rgb_loss != -1.0:
            assert not config.use_rgb_distracted_mask_for_rgb_loss_mask
            loss_collection.rgb_mask = cls._create_loss_mask_from_loss(
                loss=loss_collection.pixelwise_rgb_loss,
                percentile=config.rgb_mask_from_percentile_of_rgb_loss,
            )
        if config.normal_mask_from_percentile_of_normal_loss != -1.0:
            assert not config.use_rgb_distracted_mask_for_normal_loss_mask
            loss_collection.normal_mask = cls._create_loss_mask_from_loss(
                loss=loss_collection.get_pixelwise_normal_loss(),
                percentile=config.normal_mask_from_percentile_of_normal_loss,
            )
        if config.depth_mask_from_percentile_of_depth_loss != -1.0:
            assert not config.use_rgb_distracted_mask_for_depth_loss_mask
            loss_collection.depth_mask = cls._create_loss_mask_from_loss(
                loss=loss_collection.pixelwise_depth_loss,
                percentile=config.depth_mask_from_percentile_of_depth_loss,
            )
