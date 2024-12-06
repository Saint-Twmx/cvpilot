from measure.tool.readdicom import get_info_with_sitk_nrrd
from measure.mitral_centerline import mit_centerline
from measure.mitral_bestplane import mit_bestplane_new
from measure.mitral_cc_ap import mit_cc_ap
from measure.mitral_annulus import mit_annulus_perimeter_area
from measure.mitrial_analysis import numerical_calculation
from measure.mitral_tt import mit_tt
from measure.mitral_leaflet import mit_leaflets_length
import numpy as np
import os

from measure.post_processing_measure import post_processing_measure
def main():
    pred, head = get_info_with_sitk_nrrd(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output",
            "Demo.nii.gz"
        )
    )
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    measure = dict()

    centerline = mit_centerline(pred, simple=True)

    threeD_plane, best_plane = mit_bestplane_new(pred, centerline, measure)

    mit_annulus_perimeter_area(pred, head, threeD_plane, best_plane, measure)

    mit_cc_ap(pred, head, measure) # cc ap 只能是types = 2

    mit_tt(pred, head, best_plane, measure)  # cc ap  types 可为 1 可为 2

    mit_leaflets_length(pred, head, best_plane, measure)

    numerical_calculation(measure, pred, head)  # 指标 数值计算

    post_processing_measure(head,measure)


