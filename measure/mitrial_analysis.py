import numpy as np
import torch
from functools import reduce
import operator
from scipy.ndimage import label

def point_to_line_distance(points, point):
    assert points.shape == (2, 3), "点集应该是两个三维点"
    assert point.shape == (3,), "点应该是一个三维向量"
    A, B = points[0], points[1]
    AB = B - A
    AP = point - A
    cross_product = torch.cross(AP, AB)
    cross_norm = torch.norm(cross_product)
    AB_norm = torch.norm(AB)
    distance = cross_norm / AB_norm
    return distance.item()


def plane_angle(plane1, plane2, neg=False):
    normal_septum = plane1[:3]
    if neg:
        normal_septum = -plane1[:3]
    normal_best = plane2[:3]
    length_septum = torch.norm(normal_septum)
    length_best = torch.norm(normal_best)
    dot_product = torch.dot(normal_septum, normal_best)
    cos_angle = dot_product / (length_septum * length_best)
    angle_degrees = torch.rad2deg(torch.acos(cos_angle))
    return angle_degrees

def mit_int_septum(measure, ori_pred):

    return

def mit_non_planarity(measure):

    return

def mitral_papillary_muscle_analysis(ori_pred, measure, voxel_volume, head):

    return


def numerical_calculation(measure, ori_pred, head):

    return