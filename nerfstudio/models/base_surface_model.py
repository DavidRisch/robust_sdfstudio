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
Implementation of Base surface model.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    MultiViewLoss,
    ScaleAndShiftInvariantLoss,
    SensorDepthLoss,
    compute_scale_and_shift,
    monosdf_normal_loss,
    monosdf_normal_loss_pixelwise_l1,
    monosdf_normal_loss_pixelwise_cos,
)
from nerfstudio.model_components.patch_warping import PatchWarping
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import (
    AABBBoxCollider,
    NearFarCollider,
    SphereCollider,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils import writer

from nerfstudio.robust.loss_collection_unordered import LossCollectionUnordered
from nerfstudio.robust.loss_collection_spatial import LossCollectionSpatialBase
from nerfstudio.robust.print_utils import print_tensor_dict, print_tensor
from nerfstudio.robust.robust_loss import RobustLoss
from nerfstudio.robust.robust_loss_mask_creator import RobustLossMaskCreator
from nerfstudio.robust.log_utils import LogUtils
from nerfstudio.robust.mask_evaluator import MaskEvaluator
from nerfstudio.robust.robust_loss_mask_combiner import RobustLossMaskCombiner


@dataclass
class SurfaceModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SurfaceModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.1
    """Eikonal loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    patch_warp_loss_mult: float = 0.0
    """Multi-view consistency warping loss multiplier."""
    patch_size: int = 11
    """Multi-view consistency warping loss patch size."""
    patch_warp_angle_thres: float = 0.3
    """Threshold for valid homograph of multi-view consistency warping loss"""
    min_patch_variance: float = 0.01
    """Threshold for minimal patch variance"""
    topk: int = 4
    """Number of minimal patch consistency selected for training"""
    sensor_depth_truncation: float = 0.015
    """Sensor depth trunction, default value is 0.015 which means 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)"""
    sensor_depth_l1_loss_mult: float = 0.0
    """Sensor depth L1 loss multiplier."""
    sensor_depth_freespace_loss_mult: float = 0.0
    """Sensor depth free space loss multiplier."""
    sensor_depth_sdf_loss_mult: float = 0.0
    """Sensor depth sdf loss multiplier."""
    sparse_points_sdf_loss_mult: float = 0.0
    """sparse point sdf loss multiplier"""
    sdf_field: SDFFieldConfig = SDFFieldConfig()
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss mutliplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""
    scene_contraction_norm: Literal["inf", "l2"] = "inf"
    """Which norm to use for the scene contraction."""

    use_rgb_distracted_mask_for_rgb_loss_mask: bool = False
    use_normal_distracted_mask_for_normal_loss_mask: bool = False
    use_depth_distracted_mask_for_depth_loss_mask: bool = False

    rgb_mask_from_percentile_of_rgb_loss: float = -1.0
    """mask the rgb loss by only keeping a certain percentage of it. (0 to 100)"""
    normal_mask_from_percentile_of_normal_loss: float = -1.0
    """mask the normal loss by only keeping a certain percentage of it. (0 to 100)"""
    depth_mask_from_percentile_of_depth_loss: float = -1.0
    """mask the depth loss by only keeping a certain percentage of it. (0 to 100)"""

    robust_loss_kernel_name: str = "NoKernel"

    robust_loss_classify_patches_mode: str = "Off"

    robust_loss_combine_mode: str = "Off"

    with_detailed_eval_of_combine_mode: bool = False


class SurfaceModel(Model):
    """Base surface model

    Args:
        config: Base surface model configuration to instantiate model
    """

    config: SurfaceModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.scene_contraction_norm == "inf":
            order = float("inf")
        elif self.config.scene_contraction_norm == "l2":
            order = None
        else:
            raise ValueError("Invalid scene contraction norm")

        self.scene_contraction = SceneContraction(order=order)

        # Can we also use contraction for sdf?
        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        # Collider
        if self.scene_box.collider_type == "near_far":
            self.collider = NearFarCollider(near_plane=self.scene_box.near, far_plane=self.scene_box.far)
        elif self.scene_box.collider_type == "box":
            self.collider = AABBBoxCollider(self.scene_box, near_plane=self.scene_box.near)
        elif self.scene_box.collider_type == "sphere":
            # TODO do we also use near if the ray don't intersect with the sphere
            self.collider = SphereCollider(radius=self.scene_box.radius, soft_intersection=True)
        else:
            raise NotImplementedError

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # background model
        if self.config.background_model == "grid":
            self.field_background = TCNNNerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            )
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
            )
        else:
            # dummy background model
            self.field_background = Parameter(torch.ones(1), requires_grad=False)

        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()
        # patch warping
        self.patch_warping = PatchWarping(
            patch_size=self.config.patch_size, valid_angle_thres=self.config.patch_warp_angle_thres
        )

        # losses
        self.rgb_loss = L1Loss()
        self.rgb_loss_pixelwise = L1Loss(reduction="none")
        self.eikonal_loss = MSELoss()
        depth_loss_alpha = 0.5
        depth_loss_alpha = 0  # TODO: gradient_loss does not work with pixelweise yet
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=depth_loss_alpha, scales=1)
        self.depth_loss_pixelwise = ScaleAndShiftInvariantLoss(alpha=depth_loss_alpha, scales=1, reduction="none")
        self.patch_loss = MultiViewLoss(
            patch_size=self.config.patch_size, topk=self.config.topk, min_patch_variance=self.config.min_patch_variance
        )
        self.sensor_depth_loss = SensorDepthLoss(truncation=self.config.sensor_depth_truncation)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        self.robust_loss_mask_creator = RobustLossMaskCreator()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)
        return param_groups

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        """_summary_

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    def get_outputs(self, ray_bundle: RayBundle) -> Dict:
        # TODO make this configurable
        # compute near and far from from sphere with radius 1.0
        # ray_bundle = self.sphere_collider(ray_bundle)

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        # remove the rays that don't intersect with the surface
        # hit = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0).any(dim=1)
        # depth[~hit] = 10000.0

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # background model
        if self.config.background_model != "none":
            # TODO remove hard-coded far value
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            # merge background color to forgound color
            rgb = rgb + bg_transmittance * rgb_bg

            bg_outputs = {
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_weights": weights_bg,
            }
        else:
            bg_outputs = {}

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "ray_points": self.scene_contraction(
                ray_samples.frustums.get_start_positions()
            ),  # used for creating visiblity mask
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
        }
        outputs.update(bg_outputs)

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            points_norm = field_outputs["points_norm"]
            outputs.update({"eik_grad": grad_points, "points_norm": points_norm})

            # TODO volsdf use different point set for eikonal loss
            # grad_points = self.field.gradient(eik_points)
            # outputs.update({"eik_grad": grad_points})

            outputs.update(samples_and_field_outputs)

        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def get_outputs_flexible(self, ray_bundle: RayBundle, additional_inputs: Dict[str, TensorType]) -> Dict:
        """run the model with additional inputs such as warping or rendering from unseen rays
        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            additional_inputs: addtional inputs such as images, src_idx, src_cameras

        Returns:
            dict: information needed for compute gradients
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        outputs = self.get_outputs(ray_bundle)

        ray_samples = outputs["ray_samples"]
        field_outputs = outputs["field_outputs"]

        if self.config.patch_warp_loss_mult > 0:
            # patch warping
            warped_patches, valid_mask = self.patch_warping(
                ray_samples,
                field_outputs[FieldHeadNames.SDF],
                field_outputs[FieldHeadNames.NORMAL],
                additional_inputs["src_cameras"],
                additional_inputs["src_imgs"],
                pix_indices=additional_inputs["uv"],
            )

            outputs.update({"patches": warped_patches, "patches_valid_mask": valid_mask})

        return outputs

    def does_loss_collection_need_to_be_dense_spatial(self):
        return (
                self.config.robust_loss_kernel_name != "NoKernel" or
                self.config.robust_loss_classify_patches_mode != "Off" or
                self.config.robust_loss_combine_mode != "Off"
        )

    def get_loss_collection(self, outputs: Dict, batch: Dict, pixel_coordinates_x: TensorType[...] = None,
                            pixel_coordinates_y: TensorType[...] = None,
                            all_loss_collection_steps: Optional[Dict[str, LossCollectionBase]] = None,
                            step: Optional[int] = None) -> Union[
        LossCollectionUnordered, LossCollectionDenseSpatial]:
        loss_collection = LossCollectionUnordered()
        loss_collection.pixel_coordinates_x = pixel_coordinates_x
        loss_collection.pixel_coordinates_y = pixel_coordinates_y

        rgb_image_gt = batch["image"].to(self.device)
        rgb_image_prediction = outputs["rgb"]
        pixelwise_rgb_loss_with_channels = self.rgb_loss_pixelwise(rgb_image_gt, rgb_image_prediction)
        loss_collection.pixelwise_rgb_loss = torch.mean(pixelwise_rgb_loss_with_channels, dim=1)

        loss_collection.set_full_masks(device=self.device)  # do this early so it can be safely be overwritten later

        RobustLoss.maybe_get_loss_masks_from_distractor_mask(loss_collection=loss_collection, batch=batch,
                                                             config=self.config)

        if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
            normal_image_gt = batch["normal"].to(self.device)
            normal_image_pred = outputs["normal"]
            loss_collection.pixelwise_normal_l1 = monosdf_normal_loss_pixelwise_l1(normal_image_pred, normal_image_gt)
            loss_collection.pixelwise_normal_cos = monosdf_normal_loss_pixelwise_cos(normal_image_pred, normal_image_gt)

        if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
            # TODO check it's true that's we sample from only a single image
            # TODO only supervised pixel that hit the surface and remove hard-coded scaling for depth
            depth_image_gt = batch["depth"].to(self.device)[..., None]
            depth_image_pred = outputs["depth"]

            depth_image_gt_reshaped_scaled_and_shifted = (depth_image_gt * 50 + 0.5).reshape(1, 32, -1)
            depth_image_pred_reshaped = depth_image_pred.reshape(1, 32, -1)

            mask = torch.ones_like(depth_image_gt).reshape(1, 32, -1).bool()
            loss_collection.valid_depth_pixel_count = torch.sum(mask)
            # print("valid_depth_pixel_count", loss_collection.valid_depth_pixel_count)

            pixelwise_depth_loss = self.depth_loss_pixelwise(depth_image_pred_reshaped,
                                                             depth_image_gt_reshaped_scaled_and_shifted,
                                                             mask)
            loss_collection.pixelwise_depth_loss = pixelwise_depth_loss.flatten()

        assert isinstance(loss_collection, LossCollectionUnordered)
        self.robust_loss_mask_creator.maybe_create_loss_masks_from_losses(loss_collection=loss_collection,
                                                                          config=self.config, step=step)

        if self.does_loss_collection_need_to_be_dense_spatial():
            assert batch.get("image_is_spatial_and_contiguous", None) is True
            loss_collection: LossCollectionDenseSpatial = loss_collection.make_into_dense_spatial(
                device=self.device)

            if all_loss_collection_steps is not None:
                all_loss_collection_steps["before_kernel"] = copy.deepcopy(loss_collection)

            RobustLoss.maybe_apply_kernel_to_masks(loss_collection=loss_collection, config=self.config,
                                                   device=self.device)

            if all_loss_collection_steps is not None:
                all_loss_collection_steps["before_classify_patches"] = copy.deepcopy(loss_collection)

            RobustLoss.maybe_classify_patches(loss_collection=loss_collection, config=self.config,
                                              device=self.device)

            if all_loss_collection_steps is not None:
                all_loss_collection_steps["before_combine"] = copy.deepcopy(loss_collection)

            RobustLossMaskCombiner.maybe_combine_masks(loss_collection=loss_collection,
                                                       config=self.config,
                                                       device=self.device)

            loss_collection: LossCollectionUnordered = loss_collection.make_into_unordered()

        if all_loss_collection_steps is not None:
            all_loss_collection_steps["final"] = loss_collection

        assert isinstance(loss_collection, LossCollectionUnordered)
        return loss_collection

    def get_loss_dict(self, outputs, batch, metrics_dict=None, pixelwise=False) -> Dict:
        loss_dict = {}
        image = batch["image"].to(self.device)

        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

            # sensor depth loss
            if "sensor_depth" in batch and (
                self.config.sensor_depth_l1_loss_mult > 0.0
                or self.config.sensor_depth_freespace_loss_mult > 0.0
                or self.config.sensor_depth_sdf_loss_mult > 0.0
            ):
                l1_loss, free_space_loss, sdf_loss = self.sensor_depth_loss(batch, outputs)

                loss_dict["sensor_l1_loss"] = l1_loss * self.config.sensor_depth_l1_loss_mult
                loss_dict["sensor_freespace_loss"] = free_space_loss * self.config.sensor_depth_freespace_loss_mult
                loss_dict["sensor_sdf_loss"] = sdf_loss * self.config.sensor_depth_sdf_loss_mult

            # multi-view photoconsistency loss as Geo-NeuS
            if "patches" in outputs and self.config.patch_warp_loss_mult > 0.0:
                patches = outputs["patches"]
                patches_valid_mask = outputs["patches_valid_mask"]

                loss_dict["patch_loss"] = (
                    self.patch_loss(patches, patches_valid_mask) * self.config.patch_warp_loss_mult
                )

            # sparse points sdf loss
            if "sparse_sfm_points" in batch and self.config.sparse_points_sdf_loss_mult > 0.0:
                sparse_sfm_points = batch["sparse_sfm_points"].to(self.device)
                sparse_sfm_points_sdf = self.field.forward_geonetwork(sparse_sfm_points)[:, 0].contiguous()
                loss_dict["sparse_sfm_points_sdf_loss"] = (
                        torch.mean(torch.abs(sparse_sfm_points_sdf)) * self.config.sparse_points_sdf_loss_mult
                )

            # total variational loss for multi-resolution periodic feature volume
            if self.config.periodic_tvl_mult > 0.0:
                assert self.field.config.encoding_type == "periodic"
                loss_dict["tvl_loss"] = self.field.encoding.get_total_variation_loss() * self.config.periodic_tvl_mult

        # print_tensor_dict("loss_dict", loss_dict)

        pixel_coordinates_y, pixel_coordinates_x = None, None
        if "indices" in batch:
            pixel_coordinates_y, pixel_coordinates_x = batch["indices"][:, 1].to(self.device), batch["indices"][:,
                                                                                               2].to(self.device)

        loss_collection: Union[LossCollectionUnordered, LossCollectionDenseSpatial] = \
            self.get_loss_collection(outputs=outputs, batch=batch, pixel_coordinates_x=pixel_coordinates_x,
                                     pixel_coordinates_y=pixel_coordinates_y, step=batch["step"])

        loss_collection.apply_masks()

        loss_collection.update_dict_with_scalar_losses(loss_dict=loss_dict,
                                                       normal_loss_mult=self.config.mono_normal_loss_mult,
                                                       depth_loss_mult=self.config.mono_depth_loss_mult)

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["normal"]
        # don't need to normalize here
        # normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        normal = (normal + 1.0) / 2.0

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        if "depth" in batch:
            depth_gt = batch["depth"].to(self.device)
            depth_pred = outputs["depth"]

            # align to predicted depth and normalize
            scale, shift = compute_scale_and_shift(
                depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
            )
            depth_pred = depth_pred * scale + shift

            combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
        else:
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_depth = torch.cat([depth], dim=1)

        if "normal" in batch:
            normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "normal": combined_normal,
        }

        if "sensor_depth" in batch:
            sensor_depth = batch["sensor_depth"]
            depth_pred = outputs["depth"]

            combined_sensor_depth = torch.cat([sensor_depth[..., None], depth_pred], dim=1)
            combined_sensor_depth = colormaps.apply_depth_colormap(combined_sensor_depth)
            images_dict["sensor_depth"] = combined_sensor_depth

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict

    @torch.no_grad()
    def log_pixelwise_loss(self, ray_bundle: RayBundle, batch: Dict, step: int, log_group_name: str, image_width: int,
                           image_height: int, output_collection: OutputCollection):
        result = {}

        if "indices" in batch:
            batch["pixel_coordinates_y"], batch["pixel_coordinates_x"] = batch["indices"][:, 1], batch["indices"][:, 2]
        elif batch["image"].shape[1] == image_width and batch["image"].shape[0] == image_height:
            # this is a full image, gerate a grid of pixel coodinates
            #         batch["pixel_coordinates_y"], batch["pixel_coordinates_x"] = torch.stack(torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij"), dim=-1)
            batch["pixel_coordinates_y"], batch["pixel_coordinates_x"] = torch.meshgrid(
                torch.arange(image_height, dtype=torch.long), torch.arange(image_width, dtype=torch.long),
                indexing="ij")
        else:
            raise RuntimeError(
                "Failed to optain pixel coordinates: Batch does not contain 'indices' key and does not seem to be a full image ")

        # print("pixel_coordinates_y", batch["pixel_coordinates_y"])
        # print("pixel_coordinates_x", batch["pixel_coordinates_x"])

        # print_tensor_dict("log_pixelwise_loss batch", batch)

        relevant_key_of_batch = ["pixel_coordinates_x", "pixel_coordinates_y", "image"]
        for key in ["depth", "normal", "rgb_distracted_mask", "depth_distracted_mask", "normal_distracted_mask"]:
            if key in batch:
                relevant_key_of_batch.append(key)

        # flatten image dimensions (height, width) for easier spliting later
        ray_bundle_flattened = ray_bundle.flatten()
        if len(batch["image"].shape) == 3:
            batch_flattened = {
                key: torch.flatten(batch[key], start_dim=0, end_dim=1)
                for key in relevant_key_of_batch
            }
        elif len(batch["image"].shape) == 2:
            batch_flattened = {
                key: batch[key]
                for key in relevant_key_of_batch
            }
        else:
            assert False

        part_size = 4096
        part_count = len(ray_bundle) // part_size
        if part_size * part_count < len(ray_bundle):
            part_count += 1

        # print("part_count", part_count)

        loss_collections_by_name: Dict[str, List[LossCollectionBase]] = defaultdict(list)

        # These extract_* functions can use either original_flattened or original as input but have to return a flattened output
        def extract_part_interlaced_columns(original_flattened: Tensor, original: Tensor, part_index: int) -> Tensor:
            return original_flattened[part_index::part_count, ...]

        def extract_part_thick_rows(original_flattened: Tensor, original: Tensor, part_index: int) -> Tensor:
            return original_flattened[part_index * part_size: (part_index + 1) * part_size, ...]

        def extract_squares(original_flattened: Tensor, original: Tensor, part_index: int) -> Tensor:
            # print_tensor("original", original)

            square_edge_length = 64
            assert part_size == square_edge_length * square_edge_length
            assert len(original.shape) >= 2, original.shape
            square_count_x = original.shape[1] // square_edge_length
            square_count_y = original.shape[0] // square_edge_length
            assert square_count_x * square_edge_length == original.shape[1]
            assert square_count_y * square_edge_length == original.shape[0]

            index_x = part_index % square_count_x
            index_y = part_index // square_count_x

            part = original[index_y * square_edge_length: (index_y + 1) * square_edge_length,
                   index_x * square_edge_length: (index_x + 1) * square_edge_length, ...]

            # print_tensor("part", part)
            if isinstance(part, RayBundle):
                part = part.flatten()
            elif isinstance(part, torch.Tensor):
                part = torch.flatten(part, start_dim=0, end_dim=1)
            else:
                raise RuntimeError(f"Unexpected type: {type(part)}")
            # print_tensor("part", part)

            return part

        if self.does_loss_collection_need_to_be_dense_spatial():
            extract_part = extract_squares
        else:
            extract_part = extract_part_interlaced_columns

        for part_index in range(part_count):
            ray_bundle_part: RayBundle = extract_part(original_flattened=ray_bundle_flattened, original=ray_bundle,
                                                      part_index=part_index)
            batch_part = {
                key: extract_part(original_flattened=batch_flattened[key], original=batch[key], part_index=part_index)
                for key in relevant_key_of_batch
            }
            batch_part["image_is_spatial_and_contiguous"] = batch.get("image_is_spatial_and_contiguous", None)

            # print_tensor("ray_bundle_part.directions", ray_bundle_part.directions)
            assert len(ray_bundle_part.directions.shape) == 2 and ray_bundle_part.directions.shape[0] == part_size and \
                   ray_bundle_part.directions.shape[1] == 3, ray_bundle_part.directions.shape
            # print_tensor("batch_part pixel_coordinates_x", batch_part["pixel_coordinates_x"])
            assert len(batch_part["pixel_coordinates_x"].shape) == 1 and batch_part["pixel_coordinates_x"].shape[0] \
                   == part_size, batch_part["pixel_coordinates_x"].shape
            # print_tensor("batch_part image", batch_part["image"])
            assert len(batch_part["image"].shape) == 2 and batch_part["image"].shape[0] == part_size and \
                   batch_part["image"].shape[1] == 3, batch_part["pixel_coordinates_x"].shape

            model_outputs = self(ray_bundle_part)

            all_loss_collection_steps = {}

            self.get_loss_collection(
                outputs=model_outputs, batch=batch_part,
                pixel_coordinates_x=batch_part["pixel_coordinates_x"].to(self.device),
                pixel_coordinates_y=batch_part["pixel_coordinates_y"].to(self.device),
                all_loss_collection_steps=all_loss_collection_steps)

            for name, loss_collection in all_loss_collection_steps.items():
                loss_collection.to_device_inplace(device="cpu")
                loss_collections_by_name[name].append((loss_collection))

        unordered_loss_collection_by_name: Dict[str, LossCollectionSparseSpatial] = {}
        sparse_spatial_loss_collection_by_name: Dict[str, LossCollectionSparseSpatial] = {}

        for name, loss_collections in loss_collections_by_name.items():
            combined_loss_collection = LossCollectionUnordered.from_combination(loss_collections)

            unordered_loss_collection_by_name[name] = combined_loss_collection

            # loss_collection_dense_spatial: LossCollectionDenseSpatial = combined_loss_collection.make_into_dense_spatial(
            #     device=torch.device("cpu"))
            #
            # # print("loss_collection_dense_spatial in log_pixelwise_loss")
            # # loss_collection_dense_spatial.print_components()
            #
            # # LogUtils.log_image_with_colormap(step, log_group_names, "dense pixelwise rgb loss",
            # #                                  loss_collection_dense_spatial.pixelwise_rgb_loss)
            #
            # loss_collection_sparse_spatial: LossCollectionSparseSpatial = loss_collection_dense_spatial.make_into_sparse_spatial(
            #     image_width=image_width,
            #     image_height=image_height,
            #     device=torch.device("cpu"))

            loss_collection_sparse_spatial: LossCollectionSparseSpatial = combined_loss_collection.make_into_sparse_spatial(
                image_width=image_width,
                image_height=image_height,
                device=torch.device("cpu"))

            # print("loss_collection_sparse_spatial")
            # loss_collection_sparse_spatial.print_components()

            sparse_spatial_loss_collection_by_name[name] = loss_collection_sparse_spatial

        if "before_kernel" in sparse_spatial_loss_collection_by_name:
            self.log_pixelwise_loss_images_from_loss_collection(
                sparse_spatial_loss_collection_by_name["before_kernel"], step,
                log_group_names=[log_group_name, "10 before kernel"],
                log_losses=False, log_masks=True, log_loss_collection_ids=False, output_collection=output_collection)

        if "before_classify_patches" in sparse_spatial_loss_collection_by_name:
            self.log_pixelwise_loss_images_from_loss_collection(
                sparse_spatial_loss_collection_by_name["before_classify_patches"], step,
                log_group_names=[log_group_name, "20 before classify_patches"],
                log_losses=False, log_masks=True, log_loss_collection_ids=False, output_collection=output_collection)

        if "before_combine" in sparse_spatial_loss_collection_by_name:
            self.log_pixelwise_loss_images_from_loss_collection(
                sparse_spatial_loss_collection_by_name["before_combine"], step,
                log_group_names=[log_group_name, "30 before combine"],
                log_losses=False, log_masks=True, log_loss_collection_ids=False, output_collection=output_collection)

            # TODO: this should also be run for the normal percentile_* configuration
            if self.config.with_detailed_eval_of_combine_mode:
                if "rgb_distracted_mask" in batch:
                    original_robust_loss_combine_mode = self.config.robust_loss_combine_mode
                    log_group_id = 70
                    for robust_loss_combine_mode in ["AnyDistracted", "AllDistracted", "Majority"]:
                        self.config.robust_loss_combine_mode = robust_loss_combine_mode
                        if len(batch["rgb_distracted_mask"].shape) == 1:
                            # batch is unordered
                            mask_evaluator_loss_collection = unordered_loss_collection_by_name["before_combine"]
                        elif len(batch["rgb_distracted_mask"].shape) == 2:
                            # batch is unordered
                            mask_evaluator_loss_collection = sparse_spatial_loss_collection_by_name["before_combine"]
                        else:
                            assert False

                        mask_evaluator_loss_collection = copy.deepcopy(mask_evaluator_loss_collection)
                        mask_evaluator_loss_collection

                        RobustLossMaskCombiner.maybe_combine_masks(loss_collection=mask_evaluator_loss_collection,
                                                                   config=self.config,
                                                                   device="cpu")

                        MaskEvaluator.log_all_comparisons(loss_collection=mask_evaluator_loss_collection, batch=batch,
                                                          log_group_names=[log_group_name,
                                                                           f"{log_group_id} masks {robust_loss_combine_mode}"],
                                                          step=step,
                                                          output_collection=output_collection,
                                                          robust_loss_combine_mode=robust_loss_combine_mode)

                        log_group_id += 1

                    self.config.robust_loss_combine_mode = original_robust_loss_combine_mode

        loss_collection_sparse_spatial_final = sparse_spatial_loss_collection_by_name["final"]

        self.log_pixelwise_loss_images_from_loss_collection(
            loss_collection_sparse_spatial_final, step,
            log_group_names=[log_group_name, "80 before applying mask"],
            log_losses=True, log_masks=True, log_loss_collection_ids=True, output_collection=output_collection)
        loss_collection_sparse_spatial_final.apply_masks()
        self.log_pixelwise_loss_images_from_loss_collection(
            loss_collection_sparse_spatial_final, step,
            log_group_names=[log_group_name, "90 after applying mask"],
            log_losses=True, log_masks=False, log_loss_collection_ids=False, output_collection=output_collection)

    @torch.no_grad()
    def log_pixelwise_loss_images_from_loss_collection(self, loss_collection_spatial: LossCollectionSpatialBase,
                                                       step: int,
                                                       log_group_names: List[str],
                                                       log_losses: bool, log_masks: bool,
                                                       log_loss_collection_ids: bool,
                                                       output_collection: OutputCollection):
        result = {}

        if log_losses:
            LogUtils.log_image_with_colormap(step, log_group_names, "11 loss: rgb",
                                             loss_collection_spatial.pixelwise_rgb_loss,
                                             output_collection=output_collection)
        if log_masks:
            LogUtils.log_image_with_colormap(step, log_group_names, "31 mask: rgb",
                                             loss_collection_spatial.rgb_mask,
                                             cmap="black_and_white", output_collection=output_collection)

        # LogUtils.log_image_with_colormap(step,log_group_names,"pixelwise pixel_coordinates x", loss_collection_spatial.pixel_coordinates_x)
        # LogUtils.log_image_with_colormap(step,log_group_names,"pixelwise pixel_coordinates y", loss_collection_spatial.pixel_coordinates_y)
        if log_loss_collection_ids:
            LogUtils.log_image_with_colormap(step, log_group_names, "80 loss_collection_ids",
                                             loss_collection_spatial.loss_collection_id,
                                             output_collection=output_collection)
        if log_losses:
            LogUtils.log_image_with_colormap(step, log_group_names, "12 loss: depth",
                                             loss_collection_spatial.pixelwise_depth_loss,
                                             output_collection=output_collection)
        if log_masks:
            LogUtils.log_image_with_colormap(step, log_group_names, "32 mask: depth",
                                             loss_collection_spatial.depth_mask,
                                             cmap="black_and_white", output_collection=output_collection)

        if log_losses:
            if False:
                LogUtils.log_image_with_colormap(step, log_group_names, "14 loss: normal_l1",
                                                 loss_collection_spatial.pixelwise_normal_l1,
                                                 output_collection=output_collection)
                LogUtils.log_image_with_colormap(step, log_group_names, "15 loss: normal_cos",
                                                 loss_collection_spatial.pixelwise_normal_cos,
                                                 output_collection=output_collection)
            LogUtils.log_image_with_colormap(step, log_group_names, "13 loss: normal",
                                             loss_collection_spatial.get_pixelwise_normal_loss(),
                                             output_collection=output_collection)
        if log_masks:
            LogUtils.log_image_with_colormap(step, log_group_names, "33 mask: normal",
                                             loss_collection_spatial.normal_mask,
                                             cmap="black_and_white", output_collection=output_collection)
