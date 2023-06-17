import torch
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Literal, Any
from torchtyping import TensorType
import torch.nn.functional as F

from nerfstudio.robust.log_utils import LogUtils
from nerfstudio.utils import writer

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial
from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_unordered import LossCollectionUnordered

if TYPE_CHECKING:
    from nerfstudio.models.base_surface_model import SurfaceModelConfig


class MaskEvaluator:
    """
    Compares masks to ground truth masks.
    """

    @classmethod
    @torch.no_grad()
    def compare(cls, predicted_clean_mask: torch.FloatTensor, gt_clean_mask: torch.BoolTensor, loss_type_name: str,
                log_group_names: List[str], log_name_index: int,
                step: int):
        predicted_clean_mask = predicted_clean_mask.to("cpu")
        gt_clean_mask = gt_clean_mask.to("cpu")
        # print("predicted_clean_mask", predicted_clean_mask)
        # print("gt_clean_mask", gt_clean_mask)

        predict_distractor = (predicted_clean_mask == 0.0)
        predict_clean = (predicted_clean_mask == 1.0)
        # print("predicted", torch.sum(predict_distractor).item(), torch.sum(predict_clean).item())

        gt_distractor = torch.logical_not(gt_clean_mask)
        gt_clean = gt_clean_mask
        # print("gt", torch.sum(gt_distractor).item(), torch.sum(gt_clean).item())

        true_clean_mask = torch.logical_and(predict_clean, gt_clean)
        false_clean_mask = torch.logical_and(predict_clean, gt_distractor)
        true_distractor_mask = torch.logical_and(predict_distractor, gt_distractor)
        false_distractor_mask = torch.logical_and(predict_distractor, gt_clean)

        true_cleans = torch.sum(true_clean_mask).item()
        false_cleans = torch.sum(false_clean_mask).item()
        true_distractors = torch.sum(true_distractor_mask).item()
        false_distractors = torch.sum(false_distractor_mask).item()

        # print(f"{true_cleans=}  {false_cleans=}  {true_distractors=}  {false_distractors=}")

        total = sum((true_cleans, false_cleans, true_distractors, false_distractors))

        assert total == torch.numel(predicted_clean_mask) == torch.numel(gt_clean_mask)

        true_cleans /= total
        false_cleans /= total
        true_distractors /= total
        false_distractors /= total

        # print(f"{true_cleans=}  {false_cleans=}  {true_distractors=}  {false_distractors=}")

        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{10 + log_name_index} true_cleans: {loss_type_name}",
            scalar=true_cleans, step=step)
        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{20 + log_name_index} false_cleans: {loss_type_name}",
            scalar=false_cleans, step=step)
        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{30 + log_name_index} true_distractors: {loss_type_name}",
            scalar=true_distractors, step=step)
        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{40 + log_name_index} false_distractors: {loss_type_name}",
            scalar=false_distractors, step=step)

        if len(true_clean_mask.shape) == 2:  # only log for images, not batches
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{60 + log_name_index} true_clean_mask: {loss_type_name}",
                                             true_clean_mask, cmap="black_and_white")
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{70 + log_name_index} false_clean_mask: {loss_type_name}",
                                             false_clean_mask, cmap="black_and_white")
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{80 + log_name_index} true_distractor_mask: {loss_type_name}",
                                             true_distractor_mask, cmap="black_and_white")
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{90 + log_name_index} false_distractor_mask: {loss_type_name}",
                                             false_distractor_mask, cmap="black_and_white")

    @classmethod
    @torch.no_grad()
    def log_all_comparisons(cls, loss_collection: LossCollectionUnordered, batch: Dict[str, Any],
                            log_group_names: List[str],
                            step: int) -> None:
        # print("log_all_comparisons", log_group_names, batch.keys())
        if "rgb_distracted_mask" in batch:
            cls.compare(
                predicted_clean_mask=loss_collection.rgb_mask,
                gt_clean_mask=torch.logical_not(batch["rgb_distracted_mask"]),
                loss_type_name="rgb",
                log_group_names=log_group_names,
                log_name_index=0,
                step=step,
            )

            cls.compare(
                predicted_clean_mask=loss_collection.depth_mask,
                gt_clean_mask=torch.logical_not(batch["depth_distracted_mask"]),
                loss_type_name="depth",
                log_group_names=log_group_names,
                log_name_index=1,
                step=step,
            )

            cls.compare(
                predicted_clean_mask=loss_collection.normal_mask,
                gt_clean_mask=torch.logical_not(batch["normal_distracted_mask"]),
                loss_type_name="normal",
                log_group_names=log_group_names,
                log_name_index=2,
                step=step,
            )
