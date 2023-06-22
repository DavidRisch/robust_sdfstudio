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
    @torch.no_grad()
    def maybe_get_loss_masks_from_distractor_mask(loss_collection: LossCollectionUnordered, batch: Dict,
                                                  config: "SurfaceModelConfig") -> None:
        if "rgb_distracted_mask" in batch:
            assert len(batch["rgb_distracted_mask"].shape) == 1
            rgb_distracted_mask = batch["rgb_distracted_mask"]
            depth_distracted_mask = batch["depth_distracted_mask"]
            normal_distracted_mask = batch["normal_distracted_mask"]

            # print_tensor("apply rgb_distracted_mask before", loss_collection.rgb_mask)
            if config.use_rgb_distracted_mask_for_rgb_loss_mask:
                assert config.rgb_mask_from_percentile_of_rgb_loss == -1.0
                loss_collection.rgb_mask[rgb_distracted_mask] = 0
                # print_tensor("apply rgb_distracted_mask after", loss_collection.rgb_mask)
            if config.use_normal_distracted_mask_for_normal_loss_mask:
                assert config.rgb_mask_from_percentile_of_rgb_loss == -1.0
                loss_collection.depth_mask[depth_distracted_mask] = 0
            if config.use_depth_distracted_mask_for_depth_loss_mask:
                assert config.rgb_mask_from_percentile_of_rgb_loss == -1.0
                loss_collection.normal_mask[normal_distracted_mask] = 0

    @classmethod
    @torch.no_grad()
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
                # print_tensor(f"reshaped", old_value)

                pad_amount = size // 2
                value_with_padding = torch.nn.ReplicationPad2d((pad_amount, pad_amount, pad_amount, pad_amount))(
                    old_value)
                # print_tensor(f"value_with_padding", value_with_padding)

                modified_value = F.conv2d(input=value_with_padding, weight=kernel, stride=1, padding=0)
                modified_value = (modified_value >= 0.5).float()

                # print_tensor(f"after convolution", modified_value)
                modified_value = modified_value.reshape((height, width))
                # print_tensor(f"{attribute_name} after convolution", modified_value)

                return modified_value

            loss_collection.apply_function_to_masks(function=apply_kernel)

    @staticmethod
    @torch.no_grad()
    def _classify_patches_a(loss_collection: LossCollectionDenseSpatial, device: torch.device):
        outer_neighbourhood_kernel_size = 15
        outer_neighbourhood_kernel = torch.ones(
            (1, 1, outer_neighbourhood_kernel_size, outer_neighbourhood_kernel_size), dtype=torch.float32,
            device=device)
        # normalize so that the sum is 1
        outer_neighbourhood_kernel /= outer_neighbourhood_kernel_size ** 2

        inner_neighbourhood_kernel_size = 3
        inner_neighbourhood_kernel = torch.ones(
            (1, 1, inner_neighbourhood_kernel_size, inner_neighbourhood_kernel_size),
            dtype=torch.float32, device=device)
        # don't normalize (used to dilate mask)

        def apply_classify_patches(old_value):
            # print_tensor("apply_classify_patches old_value", old_value)

            width, height = old_value.shape[1], old_value.shape[0]
            old_value = old_value.reshape((1, 1, height, width))

            # modified_value contains values that are 0 (outlier) or 1 (inlier)
            modified_value = F.conv2d(input=old_value, weight=outer_neighbourhood_kernel, stride=1, padding="same")
            # modified_value contains values in the range [0,1] (proportion of inliers in a large neighborhood)
            # print_tensor("apply_classify_patches after first conv2d", modified_value)
            modified_value = (modified_value >= 0.6).float()
            # modified_value contains values that are 0 or 1 (large neighborhood contains enough inliers)

            # https://stackoverflow.com/a/56237377
            modified_value = torch.nn.functional.conv2d(modified_value, inner_neighbourhood_kernel, stride=1,
                                                        padding="same")
            # modified_value contains values that are 0, 1, 2, ...
            modified_value = torch.clamp(modified_value, 0, 1)
            # modified_value contains values that are 0 (outlier) or 1 (inlier)

            modified_value = modified_value.reshape((height, width))
            # print_tensor("apply_classify_patches modified_value", modified_value)

            return modified_value

        loss_collection.apply_function_to_masks(function=apply_classify_patches)

    @classmethod
    @torch.no_grad()
    def maybe_classify_patches(cls, loss_collection: LossCollectionDenseSpatial,
                               config: "SurfaceModelConfig", device: torch.device) -> None:

        if config.robust_loss_classify_patches_mode == "Off":
            return

        if config.robust_loss_classify_patches_mode == "A":
            cls._classify_patches_a(loss_collection=loss_collection, device=device)
        else:
            raise RuntimeError(
                "Unknown value for robust_loss_classify_patches_mode: " + config.robust_loss_classify_patches_mode)


