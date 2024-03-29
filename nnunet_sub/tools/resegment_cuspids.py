from typing import Tuple
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from nnunet_sub.tools.morphology import binary_dilation_3d
import time


def resegment_cuspid(
    iso_pred: np.ndarray,
    CC: Tuple,
    annulus_info: Tuple,
    kmeans: MiniBatchKMeans,
):

    n_annulus, type = annulus_info

    total_cuspid = np.stack(
        np.where((iso_pred == 7) + (iso_pred == 8) + (iso_pred == 9)), axis=-1
    )

    reseg = kmeans.predict(kmeans.normalize(total_cuspid).astype(np.int16))

    tmp_tensor = torch.zeros(size=iso_pred.shape)
    tmp_tensor[
        total_cuspid[reseg == 0][:, 0],
        total_cuspid[reseg == 0][:, 1],
        total_cuspid[reseg == 0][:, 2],
    ] = 1
    tmp_tensor[
        total_cuspid[reseg == 1][:, 0],
        total_cuspid[reseg == 1][:, 1],
        total_cuspid[reseg == 1][:, 2],
    ] = 2
    tmp_tensor[
        total_cuspid[reseg == 2][:, 0],
        total_cuspid[reseg == 2][:, 1],
        total_cuspid[reseg == 2][:, 2],
    ] = 3

    intersection = (
        binary_dilation_3d(tmp_tensor == 1, connectivity=18, iteration=3).int()
        + binary_dilation_3d(tmp_tensor == 2, connectivity=18, iteration=3).int()
        + binary_dilation_3d(tmp_tensor == 3, connectivity=18, iteration=3).int()
    ) #先给合并成一个了
    basic_region = torch.nonzero(torch.logical_and(intersection == 1, tmp_tensor != 0)) # intersection=1 且 tmp_tensor不为0的点
    intersection = torch.logical_and(intersection > 1, tmp_tensor != 0) # intersection>1 且 tmp_tensor不为0的点 和上面比就是膨胀重叠部分

    intersection = torch.nonzero(intersection != 0)

    basic_region_label = kmeans.predict(
        kmeans.normalize(basic_region).numpy().astype(np.int16)
    ) # 对basic_region每个点用之前的聚类，划分标签
    # #使用之前训练的kmeans，对basic_region，得到了basic_region_label，
    knn = KNeighborsClassifier(30, n_jobs=-1, algorithm="ball_tree")
    knn.fit(basic_region, basic_region_label) # 然后使用basic_region + basic_region_label训练了一个knn
    intersection_label = kmeans.predict(kmeans.normalize(intersection))
    count = 30
    gap = 1
    #，然后只用knn对intersection，也就是膨胀后重叠部分
    while count > 0 and gap != 0:  #迭代30次 或者 gap=0 停止
        start = time.time()
        count -= 1

        knn.fit(
            np.concatenate((basic_region, intersection), axis=0),
            np.concatenate((basic_region_label, intersection_label), axis=0),
        )
        new_intersection_label = knn.predict(intersection)
        gap = (intersection_label != new_intersection_label).sum()
        print(gap, time.time() - start)
        intersection_label = new_intersection_label

    total_cuspid = np.concatenate((basic_region, intersection), axis=0) #把有效坐标点和对应的标签堆叠起来是嘛
    reseg = np.concatenate((basic_region_label, intersection_label), axis=0)

    if n_annulus == 2:
        if type == "A":

            LCC, _, NCC = CC
            LCC = LCC.cpu().numpy()
            NCC = NCC.cpu().numpy()
            LCC_index = knn.predict(LCC.astype(np.float32))
            NCC_index = knn.predict(NCC.astype(np.float32))
            iso_pred[
                total_cuspid[reseg == LCC_index][:, 0],
                total_cuspid[reseg == LCC_index][:, 1],
                total_cuspid[reseg == LCC_index][:, 2],
            ] = 7
            iso_pred[
                total_cuspid[reseg == NCC_index][:, 0],
                total_cuspid[reseg == NCC_index][:, 1],
                total_cuspid[reseg == NCC_index][:, 2],
            ] = 9

        elif type == "B":

            LCC, RCC, _ = CC
            LCC = LCC.cpu().numpy()
            RCC = RCC.cpu().numpy()
            LCC_index = knn.predict(LCC.astype(np.float32))
            RCC_index = knn.predict(RCC.astype(np.float32))
            iso_pred[
                total_cuspid[reseg == LCC_index][:, 0],
                total_cuspid[reseg == LCC_index][:, 1],
                total_cuspid[reseg == LCC_index][:, 2],
            ] = 7
            iso_pred[
                total_cuspid[reseg == RCC_index][:, 0],
                total_cuspid[reseg == RCC_index][:, 1],
                total_cuspid[reseg == RCC_index][:, 2],
            ] = 8

        else:
            iso_pred[
                total_cuspid[reseg == 0][:, 0],
                total_cuspid[reseg == 0][:, 1],
                total_cuspid[reseg == 0][:, 2],
            ] = 7
            iso_pred[
                total_cuspid[reseg == 1][:, 0],
                total_cuspid[reseg == 1][:, 1],
                total_cuspid[reseg == 1][:, 2],
            ] = 9

    elif n_annulus == 3:
        LCC, RCC, NCC = CC
        LCC = LCC.cpu().numpy()
        RCC = RCC.cpu().numpy()
        NCC = NCC.cpu().numpy()
        LCC_index = knn.predict(LCC.astype(np.float32))
        RCC_index = knn.predict(RCC.astype(np.float32))
        NCC_index = knn.predict(NCC.astype(np.float32))
        LCC_points = total_cuspid[reseg == LCC_index]
        RCC_points = total_cuspid[reseg == RCC_index]
        NCC_points = total_cuspid[reseg == NCC_index]

        iso_pred[LCC_points[:, 0], LCC_points[:, 1], LCC_points[:, 2]] = 7
        iso_pred[RCC_points[:, 0], RCC_points[:, 1], RCC_points[:, 2]] = 8
        iso_pred[NCC_points[:, 0], NCC_points[:, 1], NCC_points[:, 2]] = 9


def resegment_cuspid_simple(kmean,iso_pred,tmp_l):
    labels = kmean.labels_
    root_point = torch.nonzero(torch.from_numpy((iso_pred == 1)))
    for l in [9,8,7,6,5,4,3,2]:
        iso_pred[iso_pred == l] = l+2
    for i in range(tmp_l):
        iso_pred[
            root_point[i][0],
            root_point[i][1],
            root_point[i][2]
        ]  = labels[i] + 1
    return iso_pred