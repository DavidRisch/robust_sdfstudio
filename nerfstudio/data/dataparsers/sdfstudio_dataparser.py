# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for friends dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType
from typing import List
import cv2

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.io import load_from_json

from nerfstudio.robust.print_utils import print_tensor_dict, print_tensor

CONSOLE = Console()


def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}


def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}


def get_sensor_depths(image_idx: int, sensor_depths):
    """function to process additional sensor depths

    Args:
        image_idx: specific image index to work with
        sensor_depths: semantics data
    """

    # sensor depth
    sensor_depth = sensor_depths[image_idx]

    return {"sensor_depth": sensor_depth}


def get_foreground_masks(image_idx: int, fg_masks):
    """function to process additional foreground_masks

    Args:
        image_idx: specific image index to work with
        fg_masks: foreground_masks
    """

    # sensor depth
    fg_mask = fg_masks[image_idx]

    return {"fg_mask": fg_mask}


def get_sparse_sfm_points(image_idx: int, sfm_points):
    """function to process additional sparse sfm points

    Args:
        image_idx: specific image index to work with
        sfm_points: sparse sfm points
    """

    # sfm points
    sparse_sfm_points = sfm_points[image_idx]
    sparse_sfm_points = BasicImages([sparse_sfm_points])
    return {"sparse_sfm_points": sparse_sfm_points}


def get_rgb_distracted_masks(image_idx: int, rgb_distracted_masks: List[np.ndarray]):
    rgb_distracted_mask = rgb_distracted_masks[image_idx]

    return {"rgb_distracted_mask": rgb_distracted_mask}


@dataclass
class SDFStudioDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: SDFStudio)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    include_sensor_depth: bool = False
    """whether or not to load sensor depth"""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    include_sfm_points: bool = False
    """whether or not to load sfm points"""
    downscale_factor: int = 1
    scene_scale: float = 2.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    pairs_sorted_ascending: Optional[bool] = True
    """if src image pairs are sorted in ascending order by similarity i.e. 
    the last element is the most similar to the first (ref)"""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    auto_orient: bool = False

    max_train_image_count: int = -1
    """maximum number of images to use from the training split. -1 to use all."""


@dataclass
class SDFStudio(DataParser):
    """SDFStudio Dataset"""

    config: SDFStudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        meta_data_path = self.config.data / f"meta_data_{split}.json"
        if not meta_data_path.is_file():
            meta_data_path = self.config.data / "meta_data.json"
        print("meta_data_path for split", split, "is", meta_data_path)
        meta = load_from_json(meta_data_path)

        indices = list(range(len(meta["frames"])))
        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]

        image_filenames = []
        depth_images = []
        normal_images = []
        sensor_depth_images = []
        foreground_mask_images = []
        sfm_points = []
        rgb_distracted_masks = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []

        frames = meta["frames"]
        if split == "train" and self.config.max_train_image_count != -1:
            print(f"Only using {self.config.max_train_image_count} images from the train split.")
            frames = frames[:self.config.max_train_image_count]

        for i, frame in enumerate(frames):
            if i not in indices:
                continue

            image_filename = self.config.data / frame["rgb_path"]

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

            if self.config.include_mono_prior:
                assert meta["has_mono_prior"]
                # load mono depth
                depth = np.load(self.config.data / frame["mono_depth_path"])
                assert len(depth.shape) == 2 and depth.shape[0] > 99 and depth.shape[1] > 99

                # Due to the weird convention of the depth (real_depth_gt = depth_image_gt * 50 + 0.5) depths can be smaller than 0 here
                assert np.min(depth) >= -0.5

                depth_images.append(torch.from_numpy(depth).float())

                # load mono normal
                normal = np.load(self.config.data / frame["mono_normal_path"])

                assert np.min(normal) >= 0 and np.max(normal) <= 1

                # transform normal to world coordinate system
                normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here

                assert len(normal.shape) == 3 and normal.shape[0] == 3 and normal.shape[1] > 99 and normal.shape[2] > 99

                # normal_value_range_by_axis = [(np.min(normal[i]), np.max(normal[i])) for i in range(3)]
                # does not always hold true in practice: assert normal_value_range_by_axis[2][1] <= 0

                normal = torch.from_numpy(normal).float()

                rot = camtoworld[:3, :3]

                normal_map = normal.reshape(3, -1)
                assert len(normal_map.shape) == 2 and normal_map.shape[0] == 3 and normal_map.shape[1] > 99
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
                assert len(normal_map.shape) == 2 and normal_map.shape[0] == 3 and normal_map.shape[1] > 99

                normal_map = rot @ normal_map
                assert len(normal_map.shape) == 2 and normal_map.shape[0] == 3 and normal_map.shape[1] > 99
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                assert len(normal_map.shape) == 3 and normal_map.shape[0] > 99 and normal_map.shape[1] > 99 and \
                       normal_map.shape[2] == 3
                normal_images.append(normal_map)

            if self.config.include_sensor_depth:
                assert meta["has_sensor_depth"]
                # load sensor depth
                sensor_depth = np.load(self.config.data / frame["sensor_depth_path"])
                sensor_depth_images.append(torch.from_numpy(sensor_depth).float())

            if self.config.include_foreground_mask:
                assert meta["has_foreground_mask"]
                # load foreground mask
                foreground_mask = np.array(Image.open(self.config.data / frame["foreground_mask"]), dtype="uint8")
                foreground_mask = foreground_mask[..., :1]
                foreground_mask_images.append(torch.from_numpy(foreground_mask).float() / 255.0)

            if self.config.include_sfm_points:
                assert meta["has_sparse_sfm_points"]
                # load sparse sfm points
                sfm_points_view = np.loadtxt(self.config.data / frame["sfm_sparse_points_view"])
                sfm_points.append(torch.from_numpy(sfm_points_view).float())

            distracted_mask_path = self.config.data / (
                frame["rgb_path"].replace("_rgb.png", "_rgb_distracted_mask.png"))
            if distracted_mask_path.is_file():
                # print("distracted_mask_path: ", distracted_mask_path)
                mask = cv2.imread(str(distracted_mask_path))
                # print("mask", mask.shape, rgb_distracted_masks)
                mask = (mask[:, :, 0] != 0)
                # print("mask", mask.shape, rgb_distracted_masks)
                mask = torch.from_numpy(mask)
                rgb_distracted_masks.append(mask)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="up",
                center_poses=False,
            )

            # we should also transform normal accordingly
            normal_images_aligned = []
            for normal_image in normal_images:
                h, w, _ = normal_image.shape
                normal_image = transform[:3, :3] @ normal_image.reshape(-1, 3).permute(1, 0)
                normal_image = normal_image.permute(1, 0).reshape(h, w, 3)
                normal_images_aligned.append(normal_image)
            normal_images = normal_images_aligned

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
            near=meta_scene_box["near"],
            far=meta_scene_box["far"],
            radius=meta_scene_box["radius"],
            collider_type=meta_scene_box["collider_type"],
        )

        height, width = meta["height"], meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        if self.config.include_mono_prior:
            additional_inputs_dict = {
                "cues": {"func": get_depths_and_normals, "kwargs": {"depths": depth_images, "normals": normal_images}}
            }
        else:
            additional_inputs_dict = {}

        if self.config.include_sensor_depth:
            additional_inputs_dict["sensor_depth"] = {
                "func": get_sensor_depths,
                "kwargs": {"sensor_depths": sensor_depth_images},
            }

        if self.config.include_foreground_mask:
            additional_inputs_dict["foreground_masks"] = {
                "func": get_foreground_masks,
                "kwargs": {"fg_masks": foreground_mask_images},
            }

        if self.config.include_sfm_points:
            additional_inputs_dict["sfm_points"] = {
                "func": get_sparse_sfm_points,
                "kwargs": {"sfm_points": sfm_points},
            }
        # load pair information
        pairs_path = self.config.data / "pairs.txt"
        if pairs_path.exists() and split == "train" and self.config.load_pairs:
            with open(pairs_path, "r") as f:
                pairs = f.readlines()
            split_ext = lambda x: x.split(".")[0]
            pairs_srcs = []
            for sources_line in pairs:
                sources_array = [int(split_ext(img_name)) for img_name in sources_line.split(" ")]
                if self.config.pairs_sorted_ascending:
                    # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
                    sources_array = [sources_array[0]] + sources_array[:1:-1]
                pairs_srcs.append(sources_array)
            pairs_srcs = torch.tensor(pairs_srcs)
            all_imgs = torch.stack(
                [get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0
            ).cuda()

            additional_inputs_dict["pairs"] = {
                "func": get_src_from_pairs,
                "kwargs": {
                    "all_imgs": all_imgs,
                    "pairs_srcs": pairs_srcs,
                    "neighbors_num": self.config.neighbors_num,
                    "neighbors_shuffle": self.config.neighbors_shuffle,
                },
            }

        if len(rgb_distracted_masks) > 0:
            print(f"loaded {len(rgb_distracted_masks)} rgb_distracted_masks")
            additional_inputs_dict["rgb_distracted_masks"] = {
                "func": get_rgb_distracted_masks,
                "kwargs": {"rgb_distracted_masks": rgb_distracted_masks},
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            depths=depth_images,
            normals=normal_images,
        )
        return dataparser_outputs
