import torch
from typing import Dict, List, Tuple, Optional
from torchtyping import TensorType

from nerfstudio.robust.print_utils import print_tensor


class LossCollectionBase:
    """
    Holders multiple components of the loss.
    The loss of every single pixel is kept separate.
    """

    def __init__(self):
        # names of all tensor attributes, they are all tensors of the same shape so many operations are done on all of them
        self.tensor_attribute_names = []

        self.pixel_coordinates_x: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("pixel_coordinates_x")
        self.pixel_coordinates_y: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("pixel_coordinates_y")

        self.loss_collection_id: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("loss_collection_id")

        self.pixelwise_rgb_loss: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("pixelwise_rgb_loss")

        self.pixelwise_depth_loss: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("pixelwise_depth_loss")

        self.valid_depth_pixel_count: Optional[int] = None

        self.pixelwise_normal_l1: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("pixelwise_normal_l1")
        self.pixelwise_normal_cos: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("pixelwise_normal_cos")

        # Tensors which control which pixels will become part of the final loss
        # -1: not part of batch
        # 0: don't include in loss
        # 1: should be part of loss
        self.rgb_mask: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("rgb_mask")
        self.depth_mask: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("depth_mask")
        self.normal_mask: Optional[TensorType[...]] = None
        self.tensor_attribute_names.append("normal_mask")

        self.masks_are_applied = False

    def set_full_masks(self, device: torch.device):
        assert self.pixelwise_rgb_loss is not None
        mask_shape = self.pixelwise_rgb_loss.shape
        fill_value = 1.0
        dtype = torch.float
        self.rgb_mask = torch.full(mask_shape, fill_value=fill_value, dtype=dtype, device=device)
        self.depth_mask = torch.full(mask_shape, fill_value=fill_value, dtype=dtype, device=device)
        self.normal_mask = torch.full(mask_shape, fill_value=fill_value, dtype=dtype, device=device)
        # print_tensor("after set_full_masks depth_mask", self.depth_mask)

    def apply_masks(self):
        assert not self.masks_are_applied

        loss_attribute_names_by_mask_names = {
            "rgb_mask": ["pixelwise_rgb_loss"],
            "depth_mask": ["pixelwise_depth_loss"],
            "normal_mask": ["pixelwise_normal_l1", "pixelwise_normal_cos"],
        }

        for mask_attribute_name, loss_attribute_names in loss_attribute_names_by_mask_names.items():
            mask = getattr(self, mask_attribute_name)
            for loss_attribute_name in loss_attribute_names:
                value = getattr(self, loss_attribute_name)
                if value is not None:
                    value[mask == 0.0] = 0
                    setattr(self, loss_attribute_name, value)

        self.masks_are_applied = True

    def get_pixelwise_normal_loss(self):
        return self.pixelwise_normal_l1 + self.pixelwise_normal_cos

    def update_dict_with_scalar_losses(self, loss_dict: Dict, normal_loss_mult: float, depth_loss_mult: float):
        if self.pixelwise_rgb_loss is not None:
            loss_dict["rgb_loss"] = torch.mean(self.pixelwise_rgb_loss)

        if self.pixelwise_depth_loss is not None:
            if self.valid_depth_pixel_count > 0:
                average_valid_depth_loss = torch.sum(self.pixelwise_depth_loss) / self.valid_depth_pixel_count
            else:
                average_valid_depth_loss = 0.0
            loss_dict["depth_loss"] = depth_loss_mult * average_valid_depth_loss

        if self.pixelwise_normal_l1 is not None:
            assert self.pixelwise_normal_cos is not None
            loss_dict["normal_loss"] = normal_loss_mult * (
                    torch.mean(self.pixelwise_normal_l1) + torch.mean(self.pixelwise_normal_cos)
            )

    def reshape_components(self, new_shape: Tuple[int, ...]):
        assert self.pixelwise_rgb_loss is not None

        for attribute_name in self.tensor_attribute_names:
            old_value = getattr(self, attribute_name)
            if old_value is not None:
                reshaped_value = torch.reshape(old_value, new_shape)
                setattr(self, attribute_name, reshaped_value)

    def to_device_inplace(self, device):
        for attribute_name in self.tensor_attribute_names:
            old_value = getattr(self, attribute_name)
            if old_value is not None:
                moved_value = old_value.to(device=device)
                setattr(self, attribute_name, moved_value)

    def print_components(self):
        print("Components of LossCollection:")
        for attribute_name in self.tensor_attribute_names:
            value = getattr(self, attribute_name)
            print_tensor(attribute_name, value, prefix="  - ")
        print(f"valid_depth_pixel_count: {self.valid_depth_pixel_count}")
