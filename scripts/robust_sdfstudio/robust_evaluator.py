#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
from rich.console import Console

from nerfstudio.pipelines.configuration_setter import ConfigurationsSetter
from nerfstudio.robust.output_collection import OutputCollection
from nerfstudio.robust.print_utils import print_tensor
from nerfstudio.utils.eval_utils import eval_setup
from scripts.robust_sdfstudio.robust_evaluator_plotter import RobustEvaluatorPlotter

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


@dataclass
class RobustEvaluator:
    """Load a checkpoint, compute some metrics, saves them to a yaml file, and saves debug images and plots."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_directory_path: Path

    max_image_count: Optional[int] = None

    def save_images(self, output_collection: OutputCollection, experiment_output_images_directory_path: Path,
                    suffix: str):
        for name, image in output_collection.images_by_name.items():
            # print_tensor(name, image)
            filename = f"{name}_{suffix}.png"
            filename = filename.replace("/", "|")
            # print(f"{filename=}")
            converter_image = (image.cpu().numpy() * 255.0).astype(np.uint8)[..., ::-1]
            cv2.imwrite(str(experiment_output_images_directory_path / Path(filename)), converter_image)

    def main(self) -> None:
        def override_config_func(current_config):
            current_config.pipeline.datamanager.eval_dataset_split_name = "train_all"

        loaded_config, loaded_pipeline, checkpoint_path = eval_setup(config_path=self.load_config,
                                                                     override_config_func=override_config_func)
        # print("loaded_config", loaded_config)

        experiment_output_directory_path = self.output_directory_path / loaded_config.experiment_name
        experiment_output_images_directory_path = experiment_output_directory_path / "images"
        experiment_output_config_path = experiment_output_directory_path / "output.yaml"

        experiment_output_directory_path.mkdir(parents=True, exist_ok=True)
        experiment_output_images_directory_path.mkdir(parents=True, exist_ok=True)

        output_dict = {
            "experiment_name": loaded_config.experiment_name,
            "method_name": loaded_config.method_name,
            "checkpoint": str(checkpoint_path),
        }

        configurations_setters = []
        configurations_setters.append(ConfigurationsSetter("default", lambda pipeline: None))

        def percentile_25(pipeline):
            pipeline.model.config.rgb_mask_from_percentile_of_rgb_loss = 25
            pipeline.model.config.normal_mask_from_percentile_of_normal_loss = 25
            pipeline.model.config.depth_mask_from_percentile_of_depth_loss = 25

        configurations_setters.append(ConfigurationsSetter("percentile_25", percentile_25))

        def percentile_50(pipeline):
            pipeline.model.config.rgb_mask_from_percentile_of_rgb_loss = 50
            pipeline.model.config.normal_mask_from_percentile_of_normal_loss = 50
            pipeline.model.config.depth_mask_from_percentile_of_depth_loss = 50

        configurations_setters.append(ConfigurationsSetter("percentile_50", percentile_50))

        def percentile_95(pipeline):
            pipeline.model.config.rgb_mask_from_percentile_of_rgb_loss = 95
            pipeline.model.config.normal_mask_from_percentile_of_normal_loss = 95
            pipeline.model.config.depth_mask_from_percentile_of_depth_loss = 95

        configurations_setters.append(ConfigurationsSetter("percentile_95", percentile_95))

        main_output_collection = OutputCollection()
        output_collections_for_configurations = [
            OutputCollection() for _ in range(len(configurations_setters))
        ]

        loaded_pipeline.robust_get_average_eval_image_metrics(
            max_image_count=self.max_image_count,
            main_output_collection=main_output_collection,
            output_collections_for_configurations=output_collections_for_configurations,
            configurations_setters=configurations_setters)

        output_dict["configurations"] = {}
        for configurations_setter, output_collection in zip(configurations_setters,
                                                            output_collections_for_configurations):
            print(f"Parsing output of configuration '{configurations_setter.name}'")
            output_for_configuration = {
                "plot_suffix": configurations_setter.name,
                "mask_evaluator_results": {},
            }
            print("output_collection.mask_evaluator_results_by_type",
                  output_collection.mask_evaluator_results_by_type.keys())
            for loss_type_name, mask_evaluator_results in output_collection.mask_evaluator_results_by_type.items():
                print("****", loss_type_name, len(mask_evaluator_results))
                true_cleans_list = []
                false_cleans_list = []
                true_distractors_list = []
                false_distractors_list = []

                for mask_evaluator_result in mask_evaluator_results:
                    true_cleans_list.append(mask_evaluator_result.true_cleans)
                    false_cleans_list.append(mask_evaluator_result.false_cleans)
                    true_distractors_list.append(mask_evaluator_result.true_distractors)
                    false_distractors_list.append(mask_evaluator_result.false_distractors)

                print(f"{true_cleans_list=}")
                print(f"{false_cleans_list=}")
                print(f"{true_distractors_list=}")
                print(f"{false_distractors_list=}")

                output_for_configuration["mask_evaluator_results"][loss_type_name] = {
                    "true_cleans_list": true_cleans_list,
                    "false_cleans_list": false_cleans_list,
                    "true_distractors_list": true_distractors_list,
                    "false_distractors_list": false_distractors_list,

                }

            print("qqqq", configurations_setter.name, output_for_configuration)
            output_dict["configurations"][configurations_setter.name] = output_for_configuration

        # Get the output and define the names to save to

        # Save output to output file
        experiment_output_config_path.write_text(yaml.dump(output_dict), "utf8")
        CONSOLE.print(f"Saved results to: {experiment_output_config_path}")

        self.save_images(output_collection=main_output_collection,
                         experiment_output_images_directory_path=experiment_output_images_directory_path,
                         suffix="default")
        for configurations_setter, output_collection in zip(configurations_setters,
                                                            output_collections_for_configurations):
            self.save_images(output_collection=output_collection,
                             experiment_output_images_directory_path=experiment_output_images_directory_path,
                             suffix=configurations_setter.name)

        CONSOLE.print(f"Saved rendering results to: {experiment_output_images_directory_path}")

        plot_directory_path: Path = experiment_output_directory_path / "plots"
        print(f"{plot_directory_path=}")

        RobustEvaluatorPlotter.plot_all(data_dict=output_dict, plot_directory_path=plot_directory_path)


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RobustEvaluator).main()


if __name__ == "__main__":
    entrypoint()
