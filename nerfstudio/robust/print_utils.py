import torch
from typing import Dict


def print_tensor(name: str, value: torch.tensor, prefix: str = "~ "):
    if value is None:
        print(f"{prefix}{name}: None")
    elif not torch.is_tensor(value):
        print(f"{prefix}{name}: {value} (not a tensor)")
    elif value.dtype == torch.float32:
        not_nan_mask = torch.logical_not(torch.isnan(value))
        value_none_nan = value[not_nan_mask]
        min_of_value = "<no values>"
        max_of_value = "<no values>"
        if len(value_none_nan) > 0:
            min_of_value = torch.min(value_none_nan)
            max_of_value = torch.max(value_none_nan)

        print(
            f"{prefix}{name}: shape: {value.shape} "
            f"min: {min_of_value} mean: {torch.nanmean(value)} max: {max_of_value}  "
            f"(dtype: {value.dtype}) "
            f"NaN: {torch.sum(torch.isnan(value))} zero: {torch.sum(value == 0.0)} other: {torch.sum(torch.logical_and(value != 0, not_nan_mask))}"
        )
    else:
        print(
            f"{prefix}{name}: shape: {value.shape} min: {torch.min(value)} max: {torch.max(value)} (dtype: {value.dtype})")


def print_tensor_dict(name: str, value: Dict[str, torch.tensor]):
    if value is None:
        print(f"$ {name}: None")
    else:
        print(f"~ {name}: {{")
        for key, element in value.items():
            print_tensor(key, element, prefix="  - ")
        print("}")
