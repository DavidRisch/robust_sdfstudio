import torch
from typing import Dict, List, Tuple, Optional
from torchtyping import TensorType

from nerfstudio.robust.print_utils import print_tensor
from nerfstudio.utils import colormaps
from nerfstudio.utils import writer
from torchtyping import TensorType


class LogUtils:

    @staticmethod
    def log_image_with_colormap(step: int, log_group_names: List[str], name: str, image: TensorType,
                                cmap: str = "viridis"):
        assert len(image.shape) == 2, image.shape
        image = image.reshape((image.shape[0], image.shape[1], 1))

        # print_tensor("log_with_colormap " + name, image)

        # Handle NaN (for float images) and -1 (for int images) sepecially
        # they mean that there is no data available for that pixel

        if image.dtype == torch.float32:
            blank_mask = torch.isnan(image)
        elif image.dtype == torch.long:
            blank_mask = (image == -1)
        elif image.dtype == torch.bool:
            blank_mask = None
        else:
            assert False, image.dtype

        # print_tensor(f"log_with_colormap {log_group_names} {name} image", image)
        # print_tensor(f"log_with_colormap {log_group_names} {name} blank_mask", blank_mask)
        image_without_blanks = torch.clone(image)
        if blank_mask is not None:
            image_without_blanks[blank_mask] = 0

        if cmap == "black_and_white":
            # print_tensor("before bool colormap image", image)
            # print_tensor("before bool colormap image_without_blanks", image_without_blanks)
            colored_loss = colormaps.apply_boolean_colormap(image_without_blanks == 1)
            # print_tensor("after bool colormap colored_loss", colored_loss)
        else:
            if torch.max(image_without_blanks) > 0:
                normalized_image = image_without_blanks / torch.max(image_without_blanks)
            else:
                normalized_image = image_without_blanks

            colored_loss = colormaps.apply_colormap(normalized_image, cmap=cmap)

        if blank_mask is not None:
            blank_color = torch.tensor([0.6, 0.6, 0.6], dtype=colored_loss.dtype)

            blank_mask = blank_mask.reshape(blank_mask.shape[:2])
            colored_loss[blank_mask] = blank_color
        # print_tensor(f"log_with_colormap {log_group_names} {name} colored_loss", colored_loss)
        writer.put_image(name="/".join(log_group_names) + "/" + name, image=colored_loss, step=step)
