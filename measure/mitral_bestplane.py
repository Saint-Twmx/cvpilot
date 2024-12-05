import torch
import numpy as np
from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array

def mit_bestplane_new(ori_pred, centerline, measure):
    '''
    Calculate the mitral annular plane
    '''

    return torch.tensor([  0.5804,   0.4436,  -0.6829, -91.5812]), torch.tensor([  0.5804,   0.4436,  -0.6829, -86.5812])



if __name__ == "__main__":
    # test
    paths = r"/mitral"
    nrrdpath = r"2053_75%_2.seg.nrrd"
    import os
    from measure.mitral_centerline import mit_centerline
    from measure.mitral_planes import mit_planes

    ori_pred, head = get_info_with_sitk_nrrd(os.path.join(paths, nrrdpath))

    centerline = mit_centerline(ori_pred, types="2")

    plane_variants_normal, mitral_point = mit_planes(ori_pred, centerline, types="2")

    projection_point_undetermined = mit_bestplane(plane_variants_normal, mitral_point, ori_pred, head)


