import numpy as np
import torch
from scipy.ndimage import binary_erosion, binary_dilation
def binary_dilation_3d(input_matrix, iterations=1):
    eroded_matrix = binary_dilation(input_matrix, iterations=iterations)
    return torch.tensor(eroded_matrix, dtype=torch.bool)

def binary_erosion_3d(input_matrix, iterations=1):
    eroded_matrix = binary_erosion(input_matrix, iterations=iterations)
    return torch.tensor(eroded_matrix, dtype=torch.bool)

def mit_annulus_perimeter_area(ori_pred, head, threeD_plane, best_plane, measure):
    valve = ori_pred.copy()
    valve[valve > 2] = 0
    valve[valve > 0] = 1
    boundary = binary_erosion_3d(valve, iterations=2)
    shell = torch.logical_xor(torch.from_numpy(valve), boundary)
    mitral_point = torch.nonzero(shell == 1)
    '''
    Calculate the mitral valve annulus
    '''
    return