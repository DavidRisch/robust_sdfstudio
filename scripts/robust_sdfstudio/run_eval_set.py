#!/usr/bin/env python


from __future__ import annotations

import os
from typing import Optional, List
import subprocess

dataset_base_path = os.environ["DATASET_BASE_DIR"]
print(f"{dataset_base_path=}")
dataset_name = "distracted_dataset_v7"
print(f"{dataset_name=}")
own_dataset_path = os.path.join(dataset_base_path, dataset_name)
print(f"{own_dataset_path=}")

repo_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
print(f"{repo_root=}")
train_script_path = os.path.join(repo_root, "scripts/train.py")
print(f"{train_script_path=}")

# this should be incremented whenever anything is changed anywhere which could change the results
eval_set_version = 6


class RunConfig:
    def __init__(self, name: str, dataset_kind: str, resolution: int,
                 use_gt_or_omnidata_maps: str,
                 sample_large_image_patches: bool,
                 use_gt_distracted_mask: bool = False,
                 robust_loss_kernel_name: str = "NoKernel",
                 simple_percentile: Optional[float] = None,
                 robust_loss_classify_patches_mode: str = "Off",
                 ):
        self.name = name
        self.dataset_kind = dataset_kind
        self.resolution = resolution
        self.use_gt_or_omnidata_maps = use_gt_or_omnidata_maps
        self.sample_large_image_patches = sample_large_image_patches
        self.use_gt_distracted_mask = use_gt_distracted_mask
        self.robust_loss_kernel_name = robust_loss_kernel_name
        self.simple_percentile = simple_percentile
        self.robust_loss_kernel_name = robust_loss_kernel_name
        self.robust_loss_classify_patches_mode = robust_loss_classify_patches_mode


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
        "--trainer.steps-per-eval-image", str(100),
        "--trainer.steps-per-save", str(1000),
        "--trainer.max_num_iterations", str(5001),
        "--pipeline.model.mono_depth_loss_mult", str(0.05),
        "--pipeline.model.mono_normal_loss_mult", str(0.05),
        "--pipeline.datamanager.sample_large_image_patches", str(run_config.sample_large_image_patches),
        "--pipeline.model.robust_loss_kernel_name", run_config.robust_loss_kernel_name,
        "--pipeline.model.robust_loss_classify_patches_mode", run_config.robust_loss_classify_patches_mode,
    ]

    if run_config.use_gt_distracted_mask:
        arguments += [
            "--pipeline.model.use_rgb_distracted_mask_for_rgb_loss_mask", str(True),
            "--pipeline.model.use_rgb_distracted_mask_for_normal_loss_mask", str(True),
            "--pipeline.model.use_rgb_distracted_mask_for_depth_loss_mask", str(True),
        ]

    if run_config.simple_percentile is not None:
        arguments += [
            "--pipeline.model.rgb_mask_from_percentile_of_rgb_loss", str(run_config.simple_percentile),
            "--pipeline.model.normal_mask_from_percentile_of_normal_loss", str(run_config.simple_percentile),
            "--pipeline.model.depth_mask_from_percentile_of_depth_loss", str(run_config.simple_percentile),
        ]

    arguments += [  # dataset arguments:
        "sdfstudio-data",
        "--data", data_path,
        "--include_mono_prior", "True",
        "--include_foreground_mask", "False",
        "--max-train-image-count", str(25),
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
                                         sample_large_image_patches=False))

    for dataset_kind in ["distracted"]:
        for resolution in [128]:
            run_configs.append(RunConfig("gtDistractedMask",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=False,
                                         use_gt_distracted_mask=True))

    for dataset_kind in ["distracted"]:
        for resolution in [128, 512]:
            run_configs.append(RunConfig("simplePercentile",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=False,
                                         simple_percentile=90.0))

    for dataset_kind in ["distracted"]:
        for resolution in [128]:
            run_configs.append(RunConfig("kernel",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=True,
                                         robust_loss_kernel_name="Box_5x5",
                                         simple_percentile=90.0))

    for dataset_kind in ["distracted"]:
        for resolution in [128]:
            run_configs.append(RunConfig("kernelPatchesA",
                                         dataset_kind=dataset_kind,
                                         resolution=resolution,
                                         use_gt_or_omnidata_maps="gt",
                                         sample_large_image_patches=True,
                                         robust_loss_kernel_name="Box_5x5",
                                         robust_loss_classify_patches_mode="A",
                                         simple_percentile=90.0))

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