import torch
from typing import Dict


def print_tensor(name: str, value: torch.tensor, prefix: str = "~ "):
    if value is None:
        print(f"{prefix}{name}: None")
    elif not torch.is_tensor(value):
        print(f"{prefix}{name}: {value} (not a tensor)")
    elif value.dtype == torch.bool:
        true_count = torch.sum(value == True)
        false_count = torch.sum(value == False)

        print(
            f"{prefix}{name}: shape: {value.shape} "
            f"true_count: {true_count} false_count: {false_count} (dtype: {value.dtype})"
        )
    elif value.dtype == torch.long:
        min_of_value = "<no values>"
        max_of_value = "<no values>"
        mean_of_value = "<no values>"
        if len(value) > 0:
            min_of_value = torch.min(value)
            max_of_value = torch.max(value)
            mean_of_value = torch.mean(value.to(dtype=torch.float32))

        print(
            f"{prefix}{name}: shape: {value.shape} "
            f"min: {min_of_value} mean: {mean_of_value} max: {max_of_value}  "
            f"(dtype: {value.dtype}) "
            f"-1: {torch.sum(value == -1)} zero: {torch.sum(value == 0)} +1: {torch.sum(value == 1)} "
            f"other: {torch.sum(torch.logical_and(value != -1, torch.logical_and(value != 0, value != 1)))}"
        )
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
            f"NaN: {torch.sum(torch.isnan(value))} -1: {torch.sum(value == -1.0)} "
            f"zero: {torch.sum(value == 0.0)} +1: {torch.sum(value == 1.0)} "
            f"other: {torch.sum(torch.sum(torch.logical_and(value != -1.0, torch.logical_and(value != 0.0, torch.logical_and(value != 1.0, not_nan_mask)))))}"
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
