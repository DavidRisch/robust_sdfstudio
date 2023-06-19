#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict

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
            plt.close(fig)

    @classmethod
    def plot_aggregated_mask_evaluator_results_part(cls, confusion_name: str, data_by_configuration_name: Dict,
                                                    pattern: str,
                                                    ax: plt.axes.Axes):
        print(f"{data_by_configuration_name=}")

        labels: List[str] = []
        values: List[float] = []

        for configuration_name, data_for_configuration_dict in data_by_configuration_name.items():
            if pattern != "" and not configuration_name.startswith(pattern):
                continue

            print(f"{data_for_configuration_dict=}")
            raw_values = data_for_configuration_dict[f"{confusion_name}s_list"]
            value = sum(raw_values) / len(raw_values)
            labels.append(configuration_name)
            values.append(value)

        ax.set_title(confusion_name)
        ax.barh(
            labels,
            values,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("average proportion of total")

    @classmethod
    def plot_aggregated_mask_evaluator_results(cls, data_by_configuration_name_by_loss_type: Dict[str, Dict[str, Any]],
                                               pattern_name: str, pattern: str,
                                               plot_directory_path: Path):

        for loss_type_name, data_by_configuration_name in data_by_configuration_name_by_loss_type.items():
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))

            for confusion_name, ax_index in [("true_clean", (0, 0)),
                                             ("false_clean", (1, 0)),
                                             ("true_distractor", (1, 1)),
                                             ("false_distractor", (0, 1))]:
                # value_list = mask_evaluator_results[f"{confusion_name}s_list"]
                cls.plot_aggregated_mask_evaluator_results_part(confusion_name=confusion_name,
                                                                data_by_configuration_name=data_by_configuration_name,
                                                                pattern=pattern,
                                                                ax=axs[ax_index[0]][ax_index[1]])

            plt.tight_layout()
            # fig.show()
            fig.savefig(plot_directory_path / f"mask_evaluator_{loss_type_name}_aggregated_{pattern_name}.png")

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

        data_by_configuration_name_by_loss_type = defaultdict(lambda: defaultdict(dict))

        for configuration_name, data_for_configuration_dict in data_dict["configurations"].items():
            cls.plot_for_configuration(data_for_configuration_dict=data_for_configuration_dict,
                                       plot_directory_path=plot_directory_path)

            for loss_type_name, mask_evaluator_results in data_for_configuration_dict["mask_evaluator_results"].items():
                data_by_configuration_name_by_loss_type[loss_type_name][configuration_name] = mask_evaluator_results

        for pattern_name, pattern in [
            ("all", ""),
            ("percentile", "percentile_"),
            ("kernel_percentile", "kernel_percentile_"),
            ("kernel_patches_percentile", "kernel_patches_percentile_"),
        ]:
            cls.plot_aggregated_mask_evaluator_results(
                data_by_configuration_name_by_loss_type=data_by_configuration_name_by_loss_type,
                pattern_name=pattern_name, pattern=pattern,
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
