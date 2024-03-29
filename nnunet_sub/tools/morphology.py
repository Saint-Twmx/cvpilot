from typing import Union

import numpy as np
import torch
from torch.nn import functional as F


def _get_3d_stru_by_connectivity(connectivity: int) -> torch.Tensor:
    stru = torch.zeros(size=(3, 3, 3), dtype=torch.float32)

    if connectivity == 6:
        stru[1, 1, 1] = 1
        stru[0, 1, 1] = 1
        stru[2, 1, 1] = 1
        stru[1, 0, 1] = 1
        stru[1, 2, 1] = 1
        stru[1, 1, 0] = 1
        stru[1, 1, 2] = 1
    elif connectivity == 18:
        stru += 1
        stru[0, 0, 0] = 0
        stru[0, 0, 2] = 0
        stru[0, 2, 0] = 0
        stru[0, 2, 2] = 0
        stru[2, 0, 0] = 0
        stru[2, 0, 2] = 0
        stru[2, 2, 0] = 0
        stru[2, 2, 2] = 0
    elif connectivity == 26:
        stru += 1
    else:
        raise NotImplementedError
    return stru.unsqueeze(dim=0).unsqueeze(dim=0)


def _get_2d_stru_by_connectivity(connectivity: int) -> torch.Tensor:
    stru = torch.zeros(size=(3, 3), dtype=torch.float32)

    if connectivity == 4:
        stru[1, 1] = 1
        stru[1, :] = 1
        stru[:, 1] = 1

    elif connectivity == 8:
        stru += 1
    else:
        raise NotImplementedError
    return stru.unsqueeze(dim=0).unsqueeze(dim=0)


def _binary_erosion_3d_tensor(
    obj: torch.Tensor,
    stru: torch.Tensor = None,
    connectivity: int = None,
    iteration: int = 1,
) -> torch.Tensor:
    ori_device = obj.device
    n_dim = len(obj.shape)

    obj_device = obj.to(device="cuda", dtype=torch.float16)
    for _ in range(5 - n_dim):
        obj_device = obj_device.unsqueeze(dim=0)

    if not ((stru is None) ^ (connectivity is None)):
        raise ValueError("One of stru or connectivity is required.")
    if connectivity is not None:
        stru = _get_3d_stru_by_connectivity(connectivity)

    for _ in range(5 - len(stru.shape)):
        stru = stru.unsqueeze(dim=0)
    stru_device = stru.to(device="cuda", dtype=torch.float16)

    for _ in range(iteration):
        conv_obj = F.conv3d(input=obj_device, weight=stru_device, padding=1)
        conv_obj = conv_obj == torch.sum(stru)

    for _ in range(5 - n_dim):
        conv_obj = conv_obj[0, :]

    return conv_obj.to(device=ori_device, dtype=torch.bool)


def _binary_erosion_2d_tensor(
    obj: torch.Tensor,
    stru: torch.Tensor = None,
    connectivity: int = None,
    iteration: int = 1,
) -> torch.Tensor:
    ori_device = obj.device
    n_dim = len(obj.shape)

    obj_device = obj.cuda().float()
    for _ in range(4 - n_dim):
        obj_device = obj_device.unsqueeze(dim=0)

    if not ((stru is None) ^ (connectivity is None)):
        raise ValueError("One of stru or connectivity is required.")
    if connectivity is not None:
        stru = _get_2d_stru_by_connectivity(connectivity)

    for _ in range(4 - len(stru.shape)):
        stru = stru.unsqueeze(dim=0)
    stru_device = stru.cuda()

    for _ in range(iteration):
        conv_obj = F.conv2d(input=obj_device, weight=stru_device, padding=1)
        conv_obj = conv_obj == torch.sum(stru)

    for _ in range(4 - n_dim):
        conv_obj = conv_obj[0, :]

    return conv_obj.to(ori_device)


def _binary_dilation_3d_tensor(
    obj: torch.Tensor, stru: torch.Tensor = None, connectivity: int = None, iteration=1
) -> torch.Tensor:

    ori_device = obj.device

    if not ((stru is None) ^ (connectivity is None)):
        raise ValueError("One of stru or connectivity is required.")
    if connectivity is not None:
        stru = _get_3d_stru_by_connectivity(connectivity)

    for _ in range(5 - len(stru.shape)):
        stru = stru.unsqueeze(dim=0)
    stru = stru.to(device="cuda", dtype=torch.float16)

    obj_device = obj.to(device="cuda", dtype=torch.float16)
    n_dim = len(obj_device.shape)

    for _ in range(5 - n_dim):
        obj_device = obj_device.unsqueeze(dim=0)

    for _ in range(iteration):
        obj_device = F.conv3d(input=obj_device, weight=stru, padding=1)
        obj_device = (obj_device >= 0.9).to(torch.float16)

    for _ in range(5 - n_dim):
        obj_device = obj_device[0, :]

    return obj_device.to(device=ori_device, dtype=torch.bool)


def _binary_dilation_2d_tensor(
    obj: torch.Tensor, stru: torch.Tensor = None, connectivity: int = None, iteration=1
) -> torch.Tensor:

    if not ((stru is None) ^ (connectivity is None)):
        raise ValueError("One of stru or connectivity is required.")
    if connectivity is not None:
        stru = _get_2d_stru_by_connectivity(connectivity)

    for _ in range(4 - len(stru.shape)):
        stru = stru.unsqueeze(dim=0)
    stru = stru.to(device="cuda")

    ori_device = obj.device
    obj_device = obj.to(device="cuda", dtype=torch.float32)
    n_dim = len(obj_device.shape)

    for _ in range(4 - n_dim):
        obj_device = obj_device.unsqueeze(dim=0)

    for _ in range(iteration):
        obj_device = F.conv2d(input=obj_device, weight=stru, padding=1)
        obj_device = (obj_device >= 0.9).to(torch.float32)

    for _ in range(4 - n_dim):
        obj_device = obj_device[0, :]

    return obj_device.to(device=ori_device, dtype=torch.bool)


def _binary_dilation_3d_array(
    obj: np.ndarray, stru: np.ndarray = None, connectivity: int = None, iteration=1
) -> torch.Tensor:

    return (
        _binary_dilation_3d_tensor(
            torch.from_numpy(obj),
            stru=(torch.from_numpy(stru) if stru is not None else None),
            connectivity=connectivity,
            iteration=iteration,
        )
        .cpu()
        .numpy()
    )


def _binary_erosion_3d_array(
    obj: np.ndarray, stru: np.ndarray = None, connectivity: int = None, iteration=1
) -> torch.Tensor:

    return (
        _binary_erosion_3d_tensor(
            torch.from_numpy(obj),
            stru=(torch.from_numpy(stru) if stru is not None else None),
            connectivity=connectivity,
            iteration=iteration,
        )
        .cpu()
        .numpy()
    )


def _binary_dilation_2d_array(
    obj: np.ndarray, stru: np.ndarray = None, connectivity: int = None, iteration=1
) -> torch.Tensor:

    return (
        _binary_dilation_2d_tensor(
            torch.from_numpy(obj),
            stru=(torch.from_numpy(stru) if stru is not None else None),
            connectivity=connectivity,
            iteration=iteration,
        )
        .cpu()
        .numpy()
    )


def _binary_erosion_2d_array(
    obj: np.ndarray, stru: np.ndarray = None, connectivity: int = None, iteration=1
) -> torch.Tensor:

    return (
        _binary_erosion_2d_tensor(
            torch.from_numpy(obj),
            stru=(torch.from_numpy(stru) if stru is not None else None),
            connectivity=connectivity,
            iteration=iteration,
        )
        .cpu()
        .numpy()
    )


def binary_dilation_3d(
    obj: Union[np.ndarray, torch.Tensor],
    stru: Union[np.ndarray, torch.Tensor] = None,
    connectivity: int = None,
    iteration=1,
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return _binary_dilation_3d_tensor(obj, stru, connectivity, iteration)
    elif isinstance(obj, np.ndarray):
        return _binary_dilation_3d_array(obj, stru, connectivity, iteration)


def binary_erosion_3d(
    obj: Union[np.ndarray, torch.Tensor],
    stru: Union[np.ndarray, torch.Tensor] = None,
    connectivity: int = None,
    iteration=1,
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return _binary_erosion_3d_tensor(obj, stru, connectivity, iteration)
    elif isinstance(obj, np.ndarray):
        return _binary_erosion_3d_array(obj, stru, connectivity, iteration)


def binary_dilation_2d(
    obj: Union[np.ndarray, torch.Tensor],
    stru: Union[np.ndarray, torch.Tensor] = None,
    connectivity: int = None,
    iteration=1,
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return _binary_dilation_2d_tensor(obj, stru, connectivity, iteration)
    elif isinstance(obj, np.ndarray):
        return _binary_dilation_2d_array(obj, stru, connectivity, iteration)


def binary_erosion_2d(
    obj: Union[np.ndarray, torch.Tensor],
    stru: Union[np.ndarray, torch.Tensor] = None,
    connectivity: int = None,
    iteration=1,
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return _binary_erosion_2d_tensor(obj, stru, connectivity, iteration)
    elif isinstance(obj, np.ndarray):
        return _binary_erosion_2d_array(obj, stru, connectivity, iteration)
