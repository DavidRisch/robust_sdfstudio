from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Literal, Any
from dataclasses import dataclass

import torch
from torchtyping import TensorType
import torch.nn.functional as F

from nerfstudio.robust.log_utils import LogUtils
from nerfstudio.robust.output_collection import OutputCollection
from nerfstudio.utils import writer

from nerfstudio.robust.loss_collection_dense_spatial import LossCollectionDenseSpatial
from nerfstudio.robust.print_utils import print_tensor

from nerfstudio.robust.loss_collection_unordered import LossCollectionUnordered

if TYPE_CHECKING:
    from nerfstudio.models.base_surface_model import SurfaceModelConfig

@dataclass(unsafe_hash=True)
class MaskEvaluatorResultKey:
    loss_type_name: str
    robust_loss_combine_mode: str

@dataclass
class MaskEvaluatorConfusionMasks:
    true_clean_mask: torch.BoolTensor
    false_clean_mask: torch.BoolTensor
    true_distractor_mask: torch.BoolTensor
    false_distractor_mask: torch.BoolTensor


@dataclass
class MaskEvaluatorResult:
    true_cleans: float
    false_cleans: float
    true_distractors: float
    false_distractors: float
    confusion_masks: Optional[MaskEvaluatorConfusionMasks]

    def add_to_global_writer(self, loss_type_name: str,
                             log_group_names: List[str], log_name_index: int,
                             step: int, robust_loss_combine_mode: str, output_collection: OutputCollection):
        output_collection.add_mask_evaluator_result(step=step,
                                                    mask_evaluator_result_key=MaskEvaluatorResultKey(
                                                        loss_type_name=loss_type_name,
                                                        robust_loss_combine_mode=robust_loss_combine_mode,
                                                    ),
                                                    mask_evaluator_result=self)

        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{10 + log_name_index} true_cleans: {loss_type_name}",
            scalar=self.true_cleans, step=step)
        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{20 + log_name_index} false_cleans: {loss_type_name}",
            scalar=self.false_cleans, step=step)
        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{30 + log_name_index} true_distractors: {loss_type_name}",
            scalar=self.true_distractors, step=step)
        writer.put_scalar(
            name="/".join(log_group_names) + "/" + f"{40 + log_name_index} false_distractors: {loss_type_name}",
            scalar=self.false_distractors, step=step)

        if self.confusion_masks:
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{60 + log_name_index} true_clean_mask: {loss_type_name}",
                                             self.confusion_masks.true_clean_mask, cmap="black_and_white",
                                             output_collection=output_collection)
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{70 + log_name_index} false_clean_mask: {loss_type_name}",
                                             self.confusion_masks.false_clean_mask, cmap="black_and_white",
                                             output_collection=output_collection)
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{80 + log_name_index} true_distractor_mask: {loss_type_name}",
                                             self.confusion_masks.true_distractor_mask, cmap="black_and_white",
                                             output_collection=output_collection)
            LogUtils.log_image_with_colormap(step, log_group_names,
                                             f"{90 + log_name_index} false_distractor_mask: {loss_type_name}",
                                             self.confusion_masks.false_distractor_mask, cmap="black_and_white",
                                             output_collection=output_collection)


class MaskEvaluator:
    """
    Compares masks to ground truth masks.
    """

    @classmethod
    @torch.no_grad()
    def compare(cls, predicted_clean_mask: torch.FloatTensor, gt_clean_mask: torch.BoolTensor):
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

        confusion_masks = MaskEvaluatorConfusionMasks(
            true_clean_mask=torch.logical_and(predict_clean, gt_clean),  # type: ignore
            false_clean_mask=torch.logical_and(predict_clean, gt_distractor),  # type: ignore
            true_distractor_mask=torch.logical_and(predict_distractor, gt_distractor),  # type: ignore
            false_distractor_mask=torch.logical_and(predict_distractor, gt_clean),  # type: ignore
        )

        true_clean_count = torch.sum(confusion_masks.true_clean_mask).item()
        false_clean_count = torch.sum(confusion_masks.false_clean_mask).item()
        true_distractor_count = torch.sum(confusion_masks.true_distractor_mask).item()
        false_distractor_count = torch.sum(confusion_masks.false_distractor_mask).item()

        # print(f"{true_cleans=}  {false_cleans=}  {true_distractors=}  {false_distractors=}")

        total = sum((true_clean_count, false_clean_count, true_distractor_count, false_distractor_count))

        assert total == torch.numel(predicted_clean_mask) == torch.numel(gt_clean_mask)

        # only log for images, not batches
        with_confusion_masks = len(confusion_masks.true_clean_mask.shape) == 2

        mask_evaluator_result = MaskEvaluatorResult(
            true_cleans=true_clean_count / total,
            false_cleans=false_clean_count / total,
            true_distractors=true_distractor_count / total,
            false_distractors=false_distractor_count / total,
            confusion_masks=confusion_masks if with_confusion_masks else None,
        )

        # print(f"{true_cleans=}  {false_cleans=}  {true_distractors=}  {false_distractors=}")

        return mask_evaluator_result

    @classmethod
    @torch.no_grad()
    def log_all_comparisons(cls, loss_collection: LossCollectionUnordered, batch: Dict[str, Any],
                            log_group_names: List[str],
                            step: int, output_collection: OutputCollection, robust_loss_combine_mode: str) -> None:
        # print("log_all_comparisons", log_group_names, batch.keys())
        if "rgb_distracted_mask" in batch:
            result_rgb = cls.compare(
                predicted_clean_mask=loss_collection.rgb_mask,  # type: ignore
                gt_clean_mask=torch.logical_not(batch["rgb_distracted_mask"]),  # type: ignore
            )
            result_rgb.add_to_global_writer(loss_type_name="rgb",
                                            log_group_names=log_group_names,
                                            log_name_index=0,
                                            step=step, robust_loss_combine_mode=robust_loss_combine_mode,
                                            output_collection=output_collection)

            result_depth = cls.compare(
                predicted_clean_mask=loss_collection.depth_mask,  # type: ignore
                gt_clean_mask=torch.logical_not(batch["depth_distracted_mask"]),  # type: ignore
            )
            result_depth.add_to_global_writer(loss_type_name="depth",
                                              log_group_names=log_group_names,
                                              log_name_index=1,
                                              step=step, robust_loss_combine_mode=robust_loss_combine_mode,
                                              output_collection=output_collection)

            result_normal = cls.compare(
                predicted_clean_mask=loss_collection.normal_mask,  # type: ignore
                gt_clean_mask=torch.logical_not(batch["normal_distracted_mask"]),  # type: ignore
            )
            result_normal.add_to_global_writer(loss_type_name="normal",
                                               log_group_names=log_group_names,
                                               log_name_index=2,
                                               step=step, robust_loss_combine_mode=robust_loss_combine_mode,
                                               output_collection=output_collection)
