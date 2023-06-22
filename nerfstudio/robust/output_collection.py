import torch
from typing import TYPE_CHECKING, Dict, List
from collections import defaultdict

if TYPE_CHECKING:
    from nerfstudio.robust.mask_evaluator import MaskEvaluatorResult, MaskEvaluatorResultKey


class OutputCollection:
    """
    Holds debug output (e.g. images) for later use (e.g. saving it to disk)
    """

    def __init__(self):
        self.images_by_name: Dict[str, torch.Tensor] = {}
        self.mask_evaluator_results_dict: Dict[MaskEvaluatorResultKey, List[MaskEvaluatorResult]] = defaultdict(list)

    def add_image(self, name: str, step: int, image: torch.Tensor):
        name = f"{name}_step{step}"

        if name in self.images_by_name:
            raise RuntimeError(f"name already exists: {name}")

        self.images_by_name[name] = image

    def add_mask_evaluator_result(self, mask_evaluator_result_key: "MaskEvaluatorResultKey", step: int,
                                  mask_evaluator_result: "MaskEvaluatorResult"):
        assert len(self.mask_evaluator_results_dict[mask_evaluator_result_key]) == step

        self.mask_evaluator_results_dict[mask_evaluator_result_key].append(mask_evaluator_result)
