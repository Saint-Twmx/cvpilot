import torch
import numpy as np
import copy

def mit_cc_ap(ori_pred, head, measure):
    data = ori_pred.copy()

    '''
    Calculate the mitral valve CC and AP indices
    '''
    return