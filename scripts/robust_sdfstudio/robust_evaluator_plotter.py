#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt

import tyro


@dataclass
class RobustEvaluatorPlotter:
    """Uses output of RobustEvaluator to create plots"""

    # Path to YAML file from RobustEvaluator.
    yaml_path: Path

    @classmethod
    def plot_mask_evaluator_result(cls, confusion_name, value_list: List[int], ax: plt.axes.Axes):
        print(f"{value_list=}")

        ax.set_title(confusion_name)
        ax.hist(
            x=value_list,
            bins=30,
            range=(0.0, 1.0)
        )
        ax.set_xlabel("proportion of total")
        ax.set_ylabel("count")

    @classmethod
    def plot_mask_evaluator_results(cls, mask_evaluator_results_by_loss_type_name: Dict[str, Any],
                                    plot_directory_path: Path, suffix: str):

        for loss_type_name, mask_evaluator_results in mask_evaluator_results_by_loss_type_name.items():
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            for confusion_name, ax_index in [("true_clean", (0, 0)),
                                             ("false_clean", (1, 0)),
                                             ("true_distractor", (1, 1)),
                                             ("false_distractor", (0, 1))]:
                value_list = mask_evaluator_results[f"{confusion_name}s_list"]
                cls.plot_mask_evaluator_result(confusion_name=confusion_name, value_list=value_list,
                                               ax=axs[ax_index[0]][ax_index[1]])

            # fig.show()
            fig.savefig(plot_directory_path / f"mask_evaluator_{loss_type_name}_{suffix}.png")

    @classmethod
    def plot_for_configuration(cls, data_for_configuration_dict: Dict[str, Any], plot_directory_path: Path) -> None:
        print("data_for_configuration_dict", data_for_configuration_dict)
        plot_directory_path.mkdir(parents=True, exist_ok=True)
        cls.plot_mask_evaluator_results(
            mask_evaluator_results_by_loss_type_name=data_for_configuration_dict["mask_evaluator_results"],
            plot_directory_path=plot_directory_path, suffix=data_for_configuration_dict["plot_suffix"])

    @classmethod
    def plot_all(cls, data_dict: Dict[str, Any], plot_directory_path: Path) -> None:
        print("data_dict", data_dict)
        for name, data_for_configuration_dict in data_dict["configurations"].items():
            cls.plot_for_configuration(data_for_configuration_dict=data_for_configuration_dict,
                                       plot_directory_path=plot_directory_path)

    def main(self) -> None:
        with open(self.yaml_path, "r") as in_file:
            data_dict = yaml.load(in_file, Loader=yaml.SafeLoader)

        plot_directory_path: Path = self.yaml_path.parent / "plots"
        print(f"{plot_directory_path=}")

        self.plot_all(data_dict=data_dict, plot_directory_path=plot_directory_path)


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RobustEvaluatorPlotter).main()


if __name__ == "__main__":
    entrypoint()
