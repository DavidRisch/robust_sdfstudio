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
            print_tensor(attribute_name, value)
        print(f"valid_depth_pixel_count: {self.valid_depth_pixel_count}")
