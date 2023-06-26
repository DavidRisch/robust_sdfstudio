import torch
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Literal
from torchtyping import TensorType
import torch.nn.functional as F

from nerfstudio.utils import writer

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial
from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_unordered import LossCollectionUnordered

if TYPE_CHECKING:
    from nerfstudio.models.base_surface_model import SurfaceModelConfig


class RobustLossMaskCreator:
    """
    Create masks based on pixelwise losses.
    """

    def __init__(self):
        # Used to collect individual pixel loss values from previous batches to get a more stable cutoff for outlier detection
        self.losses_history_by_loss_type: Dict[str, List[torch.Tensor]] = {
            loss_type_name: []  # first item is newest, last is oldest
            for loss_type_name in ("rgb", "depth", "normal")
        }

    def reset_history(self):
        # print("reset_history", len(self.losses_history_by_loss_type["rgb"]))
        for key in self.losses_history_by_loss_type:
            self.losses_history_by_loss_type[key] = []

    @torch.no_grad()
    def _create_loss_mask_from_loss(self, loss: TensorType, loss_type_name: Literal["rgb", "depth", "normal"],
                                    percentile: float, step: Optional[int]) -> torch.Tensor:
        assert 0 <= percentile <= 100
        # print_tensor("loss", loss)
        sorted_loss_values, sorted_loss_indices = torch.sort(loss)
        # print_tensor("sorted_loss_values", sorted_loss_values)
        # print_tensor("sorted_loss_indices", sorted_loss_indices)
        total_length = len(loss)

        if True:  # this block is only needed for logging to compare the new method to the old one
            # print("total_length", total_length)
            instantaneous_keep_length = int(float(total_length) * percentile / 100.0)
            # print("keep_length", keep_length)

            instantaneous_cutoff = sorted_loss_values[instantaneous_keep_length - 1].item()
            # print("step", step)
            # print("instantaneous_cutoff for", loss_type_name, instantaneous_cutoff)
            if step is not None:
                writer.put_scalar(name=f"cutoff/instantaneous: {loss_type_name} ", scalar=instantaneous_cutoff,
                                  step=step)
                instantaneous_keep_proportion = instantaneous_keep_length / total_length
                writer.put_scalar(name=f"cutoff/stable_keep_proportion: {loss_type_name} ",
                                  scalar=instantaneous_keep_proportion,
                                  step=step)

        losses_history: List[torch.Tensor] = self.losses_history_by_loss_type[loss_type_name]
        # print(f"{len(losses_history)=}")

        max_history_length = 32
        if len(losses_history) > max_history_length:
            losses_history.pop(0)
            assert len(losses_history) == max_history_length

        losses_history.insert(0, sorted_loss_values)

        loss_history = torch.cat(losses_history, dim=0)
        stable_loss_cutoff = torch.quantile(loss_history, q=percentile / 100.0, interpolation="nearest").item()

        if step is not None:
            writer.put_scalar(name=f"cutoff/stable: {loss_type_name} ", scalar=stable_loss_cutoff, step=step)

        # print("stable_loss_cutoff for", loss_type_name, stable_loss_cutoff)

        stable_keep_length = torch.searchsorted(sorted_loss_values, stable_loss_cutoff).item()
        # now sorted_loss_values[:stable_keep_length] contains all losses which are smaller than stable_loss_cutoff

        # print("stable_keep_length", stable_keep_length)

        if step is not None:
            stable_keep_proportion = stable_keep_length / total_length
            writer.put_scalar(name=f"cutoff/stable_keep_proportion: {loss_type_name} ", scalar=stable_keep_proportion,
                              step=step)

        # print("stable_keep_length for", loss_type_name, stable_keep_length)
        # if 5 < stable_keep_length < 4000:
        #     print("value at stable_keep_length for", loss_type_name, sorted_loss_values[:stable_keep_length][-1],
        #           sorted_loss_values[stable_keep_length:][0])

        # sorted_loss_values_keep = sorted_loss_values[:keep_length]
        sorted_loss_indices_keep = sorted_loss_indices[:stable_keep_length]

        # print_tensor("sorted_loss_values_keep", sorted_loss_values_keep)
        # print_tensor("sorted_loss_indices_keep", sorted_loss_indices_keep)

        mask = torch.zeros(loss.shape, dtype=torch.float, device=loss.get_device())

        mask[sorted_loss_indices_keep] = 1

        return mask

    @torch.no_grad()
    def maybe_create_loss_masks_from_losses(self, loss_collection: LossCollectionUnordered,
                                            config: "SurfaceModelConfig", step: Optional[int]) -> None:

        for _ in range(5):
            # print_tensor("apply rgb_distracted_mask before", loss_collection.rgb_mask)
            if config.rgb_mask_from_percentile_of_rgb_loss != -1.0:
                assert not config.use_rgb_distracted_mask_for_rgb_loss_mask
                loss_collection.rgb_mask = self._create_loss_mask_from_loss(
                    loss=loss_collection.pixelwise_rgb_loss, loss_type_name="rgb",
                    percentile=config.rgb_mask_from_percentile_of_rgb_loss, step=step,
                )
            if config.normal_mask_from_percentile_of_normal_loss != -1.0:
                assert not config.use_normal_distracted_mask_for_normal_loss_mask
                loss_collection.normal_mask = self._create_loss_mask_from_loss(
                    loss=loss_collection.get_pixelwise_normal_loss(), loss_type_name="normal",
                    percentile=config.normal_mask_from_percentile_of_normal_loss, step=step,
                )
            if config.depth_mask_from_percentile_of_depth_loss != -1.0:
                assert not config.use_depth_distracted_mask_for_depth_loss_mask
                loss_collection.depth_mask = self._create_loss_mask_from_loss(
                    loss=loss_collection.pixelwise_depth_loss, loss_type_name="depth",
                    percentile=config.depth_mask_from_percentile_of_depth_loss, step=step,
                )
