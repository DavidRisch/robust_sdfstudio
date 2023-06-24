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

"""
Code for sampling pixels.
"""

import random
from typing import Dict

import torch

from nerfstudio.utils.images import BasicImages
from nerfstudio.data.pixel_samplers import PixelSampler

from nerfstudio.robust.print_utils import print_tensor_dict, print_tensor


def collate_image_dataset_batch_large_patch(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # print("num_rays_per_batch", num_rays_per_batch)

    # only sample within the mask, if the mask is in the batch
    if "mask" in batch:
        raise RuntimeError("mask is not supported by PixelSamplerLargePatch")

    image_index = random.randint(0, num_images - 1)

    # TODO: set based on batch size
    patch_width = 64
    patch_height = 64

    assert patch_width == patch_height
    assert patch_width * patch_height == num_rays_per_batch

    indices_tuple = torch.meshgrid(
        torch.LongTensor([image_index]),
        torch.arange(patch_height, dtype=torch.long),
        torch.arange(patch_width, dtype=torch.long),
        indexing="ij"
    )
    indices = torch.stack(indices_tuple, dim=-1)
    indices = indices.reshape((
        num_rays_per_batch,
        3  # [image_index, y, x]
    ))

    assert image_width == image_height
    full_patch_count = image_width // patch_width
    # print(f"{full_patch_count=}")
    assert image_width == full_patch_count * patch_width
    patch_count_with_half_positions = full_patch_count + (full_patch_count - 1)
    # print(f"{patch_count_with_half_positions=}")

    patch_x_offset_index = random.randrange(0, patch_count_with_half_positions)
    patch_y_offset_index = random.randrange(0, patch_count_with_half_positions)
    patch_x_offset = patch_x_offset_index * (patch_width // 2)
    patch_y_offset = patch_y_offset_index * (patch_width // 2)

    # print(f"{patch_x_offset=} {patch_y_offset=}")

    indices[:, 1] += patch_y_offset
    indices[:, 2] += patch_x_offset

    # print_tensor("indices", indices)

    assert len(indices.shape) == 2
    assert indices.shape[0] == num_rays_per_batch, indices.shape
    assert indices.shape[1] == 3, indices.shape

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if "sparse_sfm_points" in batch:
        collated_batch["sparse_sfm_points"] = batch["sparse_sfm_points"].images[c[0]]

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    collated_batch["image_is_spatial_and_contiguous"] = True

    return collated_batch


class PixelSamplerLargePatch(PixelSampler):  # pylint: disable=too-few-public-methods
    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        assert isinstance(image_batch["image"], torch.Tensor)

        pixel_batch = collate_image_dataset_batch_large_patch(
            image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
        )

        return pixel_batch
