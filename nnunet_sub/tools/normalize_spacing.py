from typing import Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F


def _normalize_spacing_torch(
    tensor: torch.Tensor, new_shape: Tuple[int], device: torch.device, order: int = 0
):

    tensor_dim = len(tensor.shape)
    assert tensor_dim == 3 or tensor_dim == 5

    if tensor_dim == 3:
        tensor_new = torch.unsqueeze(tensor, dim=0)[None, :, :, :, :].to(device)
    else:
        tensor_new = torch.clone(tensor).to(device)

    if order == 3:
        mode = "trilinear"
    elif order == 0:
        mode = "nearest"
    else:
        raise NotImplementedError

    tensor_new = F.interpolate(
        tensor_new.to(torch.float32),
        size=tuple(new_shape.long().tolist()),
        mode=mode,
        align_corners=False if order == 3 else None,
    )

    if tensor_dim == 3:
        tensor_new = tensor_new[0, 0, :, :, :]

    return tensor_new


def _normalize_spacing_numpy(
    array: np.ndarray, new_shape: np.ndarray, device: torch.device, order: int = 0
):

    return (
        _normalize_spacing_torch(
            torch.from_numpy(array).to(device),
            torch.from_numpy(new_shape),
            device,
            order,
        )
        .cpu()
        .numpy()
    )


def normalize_spacing(
    obj: Union[np.ndarray, torch.Tensor],
    new_shape: Union[np.ndarray, torch.Tensor],
    order: int = 0,
    device: torch.device = torch.device("cuda"),
):

    assert type(obj) == type(new_shape)
    if isinstance(obj, torch.Tensor):
        return _normalize_spacing_torch(obj, new_shape, device, order)
    elif isinstance(obj, np.ndarray):
        return _normalize_spacing_numpy(obj, new_shape, device, order)
    else:
        raise NotImplementedError
