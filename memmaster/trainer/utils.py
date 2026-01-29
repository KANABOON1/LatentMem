from typing import Optional, Sequence
import torch

def merge_tensor_lists(*tensor_lists, keys=None):
    """
    Merge multiple list[list[Tensor]] into list[list[dict[str, Tensor]]].
    
    Args:
        *tensor_lists: any number of list[list[Tensor]]
        keys: list of keys to use in the dict; if None, use str index ("t0", "t1", ...)
    
    Returns:
        list[list[dict[str, Tensor]]]
    """
    num_lists = len(tensor_lists)
    
    # Generate default keys if not provided
    if keys is None:
        keys = [f"t{i}" for i in range(num_lists)]
    elif len(keys) != num_lists:
        raise ValueError(f"Expected {num_lists} keys, but got {len(keys)}")
    
    # Merge
    return [
        [
            {k: t for k, t in zip(keys, tensors)}
            for tensors in zip(*inner_lists)
        ]
        for inner_lists in zip(*tensor_lists)
    ]

def concat_merged_list(merged_list):
    """
    Concatenate tensors from a merged list[list[dict[str, Tensor]]] along the first dimension.

    Args:
        merged_list: list[list[dict[str, Tensor]]], output of merge_tensor_lists

    Returns:
        dict[str, Tensor]: each key contains the concatenated tensor
    """
    if not merged_list:
        return {}

    keys = merged_list[0][0].keys()

    key_to_tensors = {k: [] for k in keys}

    for inner_list in merged_list:      
        for d in inner_list:             
            for k, t in d.items():
                key_to_tensors[k].append(t)

    key_to_concat = {k: torch.cat(v, dim=0) for k, v in key_to_tensors.items()}

    return key_to_concat


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])
