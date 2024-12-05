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
    # voxel_volume = reduce(operator.mul, head['spacing']) # 立方毫米
    #
    # zxs_points = torch.nonzero(torch.from_numpy(ori_pred==16))
    # zxs_voxels = len(zxs_points)
    # zxs_volume = zxs_voxels * voxel_volume
    # measure["left_ventricular_volume"] = zxs_volume/1000  #左心室体积 容积
    #
    # zxs_plane_dis = calculate_points_plane_distance(zxs_points.cuda(), measure["best_plane"].cuda())
    # _, max_index = torch.max(torch.abs(zxs_plane_dis),dim=0)
    # points_phy = convert_to_physical_coordinates_gpu(
    #     torch.stack((zxs_points[max_index], project_points_onto_plane_gpu(
    #         zxs_points[max_index].unsqueeze(0), measure["best_plane"]).cpu()[0]),
    #                 dim=0), head['spacing'])  # 左室心尖距离瓣环平面距离
    # measure["left_ventricle_apex_to_mitral_valve_annular_distance"] = np.linalg.norm(points_phy[0] - points_phy[-1])
    #
    # zxj_voxels = len(torch.nonzero(torch.from_numpy(ori_pred==17)))
    # zxj_volume = zxj_voxels * voxel_volume
    # measure["left_ventricular_myocardial_volume"] = zxj_volume/1000 # 左心肌 体积
    #
    # mitral_papillary_muscle_analysis(ori_pred, measure, voxel_volume, head)  # 乳头肌分析
    #
    # mit_int_septum(measure, ori_pred)  #房间隔 与 瓣环平面 夹角
    #
    # mit_non_planarity(measure)      # 非平面化角度
    #
    # A2 = measure["A2_points_curve_dis"]
    # P2 = measure["P2_points_curve_dis"]
    # AP = measure["mitral_ap"]
    # CC = measure["mitral_cc_real"]
    # AH = measure["mitral_annulus_hight"]
    # measure["leaflet_to_annulus_ratio"] = (A2 + P2) / AP# 瓣叶 - 瓣环指数 =（A2+P2）/瓣环前后径
    # measure["coaptation_index"] = (A2 + P2 - AP) / 2 # 对合指数=（前叶+后叶- 瓣环前后径）/2
    # measure["AHCWR"] = AH / CC # 二尖瓣环高度与连合宽度比（AHCWR）
    #
    # #主动脉瓣环 - 二尖瓣瓣环  夹角
    # zdm = torch.nonzero(torch.from_numpy(ori_pred == 11))
    # zdm_by_dis = calculate_points_distance_torch(
    #     torch.mean(torch.nonzero(
    #         torch.from_numpy((ori_pred == 5)|(ori_pred == 6)|(ori_pred == 7))).type(torch.float),
    #                dim=0).unsqueeze(0), zdm)
    # zdm_point = zdm[zdm_by_dis[0] < 55]
    # zxs_cen_point = torch.mean(zxs_points.type(torch.float),dim=0)
    # zdm_point_plane = fit_plane_pca(zdm_point.type(torch.float))
    # zxs_zdmplane_dis = calculate_points_plane_distance(zxs_cen_point.unsqueeze(0), zdm_point_plane)
    # if zxs_zdmplane_dis[0] > 0:
    #     zdm_point_plane = torch.cat([-zdm_point_plane[:3], zdm_point_plane[3].unsqueeze(0)])
    # step = 1
    # while step < 100:
    #     zdm_point_plane[-1] += 5
    #     zdm_plane_dis = calculate_points_plane_distance(zdm_point, zdm_point_plane)
    #     if torch.sum(zdm_plane_dis < 0).item() / len(zdm_point) < 0.01:
    #         break
    #     step+=1
    #
    # measure["aortic_valve_annulus_and_mitral_valve_annulus_angle"] = float(plane_angle(zdm_point_plane, measure["best_plane"]))

    return