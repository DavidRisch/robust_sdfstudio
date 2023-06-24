#!/usr/bin/env python


from __future__ import annotations

import os
from typing import Optional, List
import subprocess

from scripts.robust_sdfstudio.robust_config import RobustConfig

dataset_base_path = os.environ["DATASET_BASE_DIR"]
print(f"{dataset_base_path=}")
dataset_name = "distracted_dataset_v8"
print(f"{dataset_name=}")
own_dataset_path = os.path.join(dataset_base_path, dataset_name)
print(f"{own_dataset_path=}")

repo_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
print(f"{repo_root=}")
train_script_path = os.path.join(repo_root, "scripts/train.py")
print(f"{train_script_path=}")

# this should be incremented whenever anything is changed anywhere which could change the results
eval_set_version = 9


class RunConfig:
    def __init__(self, name: str, dataset_kind: str, resolution: int,
                 use_gt_or_omnidata_maps: str,
                 sample_large_image_patches: bool,
                 depth_loss_name: str,
                 robust_config: RobustConfig,
                 ):
        self.name = name
        self.dataset_kind = dataset_kind
        self.resolution = resolution
        self.use_gt_or_omnidata_maps = use_gt_or_omnidata_maps
        self.sample_large_image_patches = sample_large_image_patches
        self.depth_loss_name = depth_loss_name
        self.robust_config = robust_config


def prepare_run(run_config: RunConfig) -> List[str]:
    experiment_name = f"eval_set-{eval_set_version}-{run_config.name}-{run_config.dataset_kind}-{run_config.resolution}"

    data_path = os.path.join(own_dataset_path, "suzanne", str(run_config.resolution), run_config.dataset_kind)

    arguments = [
        "python",
        train_script_path,
        "neus-facto",
        "--pipeline.model.sdf-field.inside-outside", "False",
        "--vis", "wandb",
        "--experiment-name", experiment_name,
        # "--logging.local-writer.max-log-size", "0",
        "--pipeline.datamanager.train_num_rays_per_batch", str(4096),
        "--pipeline.datamanager.eval_num_rays_per_batch", str(4096),
        "--trainer.steps-per-eval-image", str(500),
        "--trainer.steps-per-save", str(1000),
        "--trainer.max_num_iterations", str(10001),
        "--pipeline.model.mono_depth_loss_mult", str(0.05),
        "--pipeline.model.mono_normal_loss_mult", str(0.05),
        "--pipeline.datamanager.sample_large_image_patches", str(run_config.sample_large_image_patches),
        "--pipeline.model.robust_loss_kernel_name", run_config.robust_config.robust_loss_kernel_name,
        "--pipeline.model.robust_loss_classify_patches_mode",
        run_config.robust_config.robust_loss_classify_patches_mode,
        "--pipeline.model.depth_loss_name",
        run_config.depth_loss_name,
    ]

    if run_config.robust_config.use_gt_distracted_mask:
        arguments += [
            "--pipeline.model.use_rgb_distracted_mask_for_rgb_loss_mask", str(True),
            "--pipeline.model.use_normal_distracted_mask_for_normal_loss_mask", str(True),
            "--pipeline.model.use_depth_distracted_mask_for_depth_loss_mask", str(True),
        ]

    if run_config.robust_config.simple_percentile is not None:
        arguments += [
            "--pipeline.model.rgb_mask_from_percentile_of_rgb_loss", str(run_config.robust_config.simple_percentile),
            "--pipeline.model.normal_mask_from_percentile_of_normal_loss",
            str(run_config.robust_config.simple_percentile),
            "--pipeline.model.depth_mask_from_percentile_of_depth_loss",
            str(run_config.robust_config.simple_percentile),
        ]

    arguments += [  # dataset arguments:
        "sdfstudio-data",
        "--data", data_path,
        "--include_mono_prior", "True",
        "--include_foreground_mask", "False",
        "--max-train-image-count", str(60),
        "--use_gt_or_omnidata_maps", run_config.use_gt_or_omnidata_maps
    ]

    # print("arguments_list", arguments)
    print("arguments:")
    print(" ".join(arguments))

    return arguments


def execute_run(arguments: List[str]):
    print("execute_run with arguments:")
    print(" ".join(arguments))
    result = subprocess.run(arguments)
    print(f"{result}")
    assert result.returncode == 0


def main():
    run_configs = []

    for dataset_kind in ["clean", "distracted"]:
        for resolution in [512]:
            run_configs.append(RunConfig("baseline",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=False,
                                         depth_loss_name="L1Loss",
                                         robust_config=RobustConfig(
                                         )))
    if False:
        for dataset_kind in ["distracted"]:
            for resolution in [128]:
                run_configs.append(RunConfig("gtDistractedMask",
                                             dataset_kind=dataset_kind,
                                             resolution=resolution,
                                             use_gt_or_omnidata_maps="gt",
                                             sample_large_image_patches=False,
                                             depth_loss_name="L1Loss",
                                             robust_config=RobustConfig(
                                                 use_gt_distracted_mask=True
                                             )))

    simple_percentile = 75.0

    for dataset_kind in ["distracted"]:
        for resolution in [128, 512]:
            run_configs.append(RunConfig("simplePercentile",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=False,
                                         depth_loss_name="L1Loss",
                                         robust_config=RobustConfig(
                                             simple_percentile=simple_percentile
                                         )))

    for dataset_kind in ["distracted"]:
        for resolution in [128]:
            run_configs.append(RunConfig("kernel",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=True,
                                         depth_loss_name="L1Loss",
                                         robust_config=RobustConfig(
                                             robust_loss_kernel_name="Box_5x5",
                                             simple_percentile=simple_percentile
                                         )))

    for dataset_kind in ["distracted"]:
        for resolution in [128, 512]:
            run_configs.append(RunConfig("kernelPatchesA",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=True,
                                         depth_loss_name="L1Loss",
                                         robust_config=RobustConfig(
                                             robust_loss_kernel_name="Box_5x5",
                                             robust_loss_classify_patches_mode="A",
                                             simple_percentile=simple_percentile
                                         )))

    run_arguments: List[List[str]] = []
    for run_config in run_configs:
        arguments = prepare_run(run_config)
        run_arguments.append(arguments)

    run_arguments_with_index = [
        (index, arguments) for (index, arguments) in enumerate(run_arguments)
    ]

    # run_arguments_with_index = run_arguments_with_index[3:]

    for index, arguments in run_arguments_with_index:
        print()
        print("run index: ", index)
        execute_run(arguments)


if __name__ == "__main__":
    main()
