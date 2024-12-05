import numpy as np
from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array
from measure.tool.spendtime import log_time
import torch

@log_time
def mit_centerline(ori_pred, simple=False):
    iso_pred = ori_pred[::2, ::2, ::2].copy()  # 变小
    for table in [1, 2, 3, 4, 5, 6, 7, 8, 10, 16]:
        iso_pred[iso_pred == table] = 66
    iso_pred[iso_pred < 66] = 0
    iso_pred[iso_pred == 66] = 1
    Apoint_landmark = torch.nonzero(torch.from_numpy(ori_pred == 17)).type(torch.float32).mean(dim=0)/2
    Bpoint_landmark = torch.nonzero(torch.from_numpy(ori_pred == 10)).type(torch.float32).mean(dim=0)/2
    if simple:
        centerline = torch.stack([Apoint_landmark*2 , Bpoint_landmark*2], dim=0)
    else:
        pass
    return centerline


if __name__ == "__main__":
    # test
    paths = r"/mitral"
    nrrdpath = r"000002_63%_1个瓣.seg.nrrd"
    import os
    data, head = get_info_with_sitk_nrrd(os.path.join(paths, nrrdpath))
    centerline = mit_centerline(data, types="1")