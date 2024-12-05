import torch
import numpy as np
import copy


def mit_tt(ori_pred, head, best_plane, measure):
    '''
    Calculate the mitral valve TT index
    '''
    return