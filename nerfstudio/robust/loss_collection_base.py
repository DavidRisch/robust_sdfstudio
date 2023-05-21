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
        self.pixel_coordinates_x: Optional[TensorType[...]] = None
        self.pixel_coordinates_y: Optional[TensorType[...]] = None

        self.loss_collection_id: Optional[TensorType[...]] = None

        self.pixelwise_rgb_loss: Optional[TensorType[...]] = None

        self.pixelwise_depth_loss: Optional[TensorType[...]] = None
        self.valid_depth_pixel_count: Optional[TensorType[...]] = None

        self.pixelwise_normal_l1: Optional[TensorType[...]] = None
        self.pixelwise_normal_cos: Optional[TensorType[...]] = None

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

        if self.pixel_coordinates_x is not None:
            self.pixel_coordinates_x = torch.reshape(self.pixel_coordinates_x, new_shape)
        if self.pixel_coordinates_y is not None:
            self.pixel_coordinates_y = torch.reshape(self.pixel_coordinates_y, new_shape)

        if self.loss_collection_id is not None:
            self.loss_collection_id = torch.reshape(self.loss_collection_id, new_shape)

        if self.pixelwise_rgb_loss is not None:
            self.pixelwise_rgb_loss = torch.reshape(self.pixelwise_rgb_loss, new_shape)

        if self.pixelwise_depth_loss is not None:
            self.pixelwise_depth_loss = torch.reshape(self.pixelwise_depth_loss, new_shape)

        if self.pixelwise_normal_l1 is not None:
            self.pixelwise_normal_l1 = torch.reshape(self.pixelwise_normal_l1, new_shape)
        if self.pixelwise_normal_cos is not None:
            self.pixelwise_normal_cos = torch.reshape(self.pixelwise_normal_cos, new_shape)

    def to_device_inplace(self, device):
        if self.pixel_coordinates_x is not None:
            self.pixel_coordinates_x = self.pixel_coordinates_x.to(device=device)

        if self.pixel_coordinates_y is not None:
            self.pixel_coordinates_y = self.pixel_coordinates_y.to(device=device)

        if self.loss_collection_id is not None:
            self.loss_collection_id = self.loss_collection_id.to(device=device)

        if self.pixelwise_rgb_loss is not None:
            self.pixelwise_rgb_loss = self.pixelwise_rgb_loss.to(device=device)

        if self.pixelwise_depth_loss is not None:
            self.pixelwise_depth_loss = self.pixelwise_depth_loss.to(device=device)

        # if self.valid_depth_pixel_count is not None:
        #     self.valid_depth_pixel_count = self.valid_depth_pixel_count.to(device=device)

        if self.pixelwise_normal_l1 is not None:
            self.pixelwise_normal_l1 = self.pixelwise_normal_l1.to(device=device)

        if self.pixelwise_normal_cos is not None:
            self.pixelwise_normal_cos = self.pixelwise_normal_cos.to(device=device)

    def print_components(self):
        print("Components of LossCollection:")
        print_tensor("pixel_coordinates_x", self.pixel_coordinates_x)
        print_tensor("pixel_coordinates_y", self.pixel_coordinates_y)
        print_tensor("pixelwise_rgb_loss", self.pixelwise_rgb_loss)
        print_tensor("pixelwise_depth_loss", self.pixelwise_depth_loss)
        print_tensor("pixelwise_normal_l1", self.pixelwise_normal_l1)
        print_tensor("pixelwise_normal_cos", self.pixelwise_normal_cos)
