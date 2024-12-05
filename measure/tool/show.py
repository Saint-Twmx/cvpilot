import numpy as np
import os
from measure.tool.readdicom import handle_save_array, get_info_with_sitk_nrrd

def get_some_nrrd(measure, ori_pred, head):
    test, infor = get_info_with_sitk_nrrd(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "output",
            "check.seg.nrrd"
        )
    )


    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output",
        "check.seg.nrrd"
    )


    handle_save_array(save_path,test,head)

