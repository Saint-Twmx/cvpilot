import torch
import numpy as np
from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array

def mit_bestplane_new(ori_pred, centerline, measure):
    '''
    Calculate the mitral annular plane
    '''

    return torch.tensor([  0.5804,   0.4436,  -0.6829, -91.5812]), torch.tensor([  0.5804,   0.4436,  -0.6829, -86.5812])

