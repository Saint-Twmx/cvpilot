import numpy
import numpy as np
import torch
from scipy.ndimage import label
from measure.tool.spendtime import log_time

@ log_time
def postprocess(pred: numpy.ndarray):
    data = np.array(pred.clone())
    zdm = (data == 11).astype(int)
    labeled_array, num_features = label(zdm)
    if num_features>1:
        sizes = np.bincount(labeled_array.ravel())
        lst = list(sizes[1:])
        max_index = lst.index(max(lst))
        second_max_index = lst.index(max([x for i, x in enumerate(lst) if i != max_index]))
        tag_1_point = torch.mean(torch.nonzero(torch.from_numpy((labeled_array == max_index + 1))).type(torch.float),dim=0)
        tag_2_point = torch.mean(torch.nonzero(torch.from_numpy((labeled_array == second_max_index + 1))).type(torch.float), dim=0)
        tag_xg_point = torch.mean(torch.nonzero((pred == 5)|(pred == 6)|(pred == 7)).type(torch.float), dim=0)
        if lst[second_max_index] < 5000:
            Tag = max_index + 1
        elif torch.norm(tag_1_point-tag_xg_point) > torch.norm(tag_2_point-tag_xg_point):
            Tag = second_max_index + 1
        else:
            Tag = max_index + 1
        clear_points = torch.nonzero(torch.from_numpy((labeled_array != Tag)&((labeled_array != 0))))
        pred[clear_points[:, 0], clear_points[:, 1], clear_points[:, 2]] = 0

    return pred
