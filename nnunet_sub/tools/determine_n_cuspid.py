from typing import Tuple, List
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from nnunet_sub.tools.resegment_cuspids import resegment_cuspid_simple

def calculate_points_distance_torch(
    pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    pts1 = pts1.to(dtype=torch.float64)
    pts2 = pts2.to(dtype=torch.float64)
    return torch.sqrt_(
        torch.sum(torch.pow(pts1[:, None, :] - pts2[None, :, :], 2), dim=-1)
    )

class KMeansNormalization:
    def __init__(self, mean: np.ndarray):
        self.mean = mean

    def __call__(self, X: np.ndarray):
        return X - self.mean

    def revert(self, X: np.ndarray):
        return X + self.mean


def determine_n_of_cuspid(
    root_points: torch.Tensor, kmean_norm: KMeansNormalization, n_points: int
) -> Tuple[int, MiniBatchKMeans]:

    kmeans = [
        MiniBatchKMeans(n_clusters=2, batch_size=n_points, random_state=0, n_init=1),
        MiniBatchKMeans(n_clusters=3, batch_size=n_points, random_state=0, n_init=1),
        MiniBatchKMeans(n_clusters=4, batch_size=n_points, random_state=0, n_init=1),
    ]
    norm_points = (kmean_norm(root_points)).round_().cpu().numpy().astype(np.int16)

    test = []
    for m in range(len(kmeans)):
        test.append(kmeans[m].fit(norm_points))
    kmeans: List[MiniBatchKMeans] = test

    random_idx = np.random.random_integers(
        low=0, high=norm_points.shape[0] - 1, size=(n_points)
    )

    test = []
    for i in range(len(kmeans)):
        test.append(silhouette_score(norm_points[random_idx, :], kmeans[i].labels_[random_idx]))
    cuspid_score = test

    n_annulus = int(np.argmax(cuspid_score))
    setattr(kmeans[n_annulus], "normalize", kmean_norm)
    return n_annulus + 2, kmeans[n_annulus]



def resegment_cuspid(iso_pred):
    '''
    Call the determine_n_of_cuspid method to perform clustering on the midnight data, and readjust the leaflet labels.
    '''

    root_point = torch.nonzero(torch.from_numpy((iso_pred == 1)))
    aro_points = torch.nonzero(torch.from_numpy((iso_pred == 7)))
    dis = calculate_points_distance_torch(torch.mean(root_point.type(torch.float32),dim=0).unsqueeze(0),
                                          aro_points)
    values_below_threshold = (dis < torch.mean(dis)/3.5)
    need_aro_points = aro_points[values_below_threshold[0]]
    tmp_l = len(root_point)
    root_point =  torch.cat((root_point, need_aro_points))

    print(
        "You should adjust the root_point range and try to carry a part of the aorta so that the clustering can be done more morphologically."
    )
    kmean_norm = KMeansNormalization(
        torch.mean(root_point.type(torch.float32), dim=0, keepdims=True).cpu().numpy()
    )
    n_annulus, kmean = determine_n_of_cuspid(root_point, kmean_norm, 1024)
    print(
        "the three mini-batch K-means clustering algorithms (k = 2,3) was performed to evaluate the number of cuspids."
    )
    # if n_annulus == 2:
    #     type, LCC, RCC, NCC, cuspid_floor, annulus_plane_params = determine_bicuspid(
    #     )
    #
    # elif n_annulus == 3:
    #     type, LCC, RCC, NCC, cuspid_floor, annulus_plane_params = determine_tricuspid(
    #     )
    #
    # elif n_annulus == 4:
    #     raise NotImplementedError("Quad cuspid found.")
    # else:
    #     raise NotImplementedError

    print(
        "The information of cuspids type and Sinotubular junction were obtained by clustering results."
    )

    # resegment_cuspids(
    #     iso_pred.cpu().numpy(),
    #     (LCC, RCC, NCC),
    #     (n_annulus, type),
    #     kmean,
    #     image_info,
    # )
    iso_pred = resegment_cuspid_simple(kmean,iso_pred,tmp_l)

    print(
        "Please be aware that the code provides a framework method for the three steps mentioned above; you will need to customize the method calls to align the segmentation results with the actual objectives."
    ,"Only the most concise partition scheme is used here.")
    return iso_pred