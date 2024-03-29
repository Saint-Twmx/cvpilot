import logging
import os
import sys
from typing import NoReturn
import numpy as np
import torch
import SimpleITK as sitk
from nnunet_sub.tools.normalize_spacing import normalize_spacing
from nnunet_sub.tools.determine_n_cuspid import resegment_cuspid
curdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, curdir)

# # 设置默认的GPU设备
# torch.cuda.set_device(1)

if __package__:
    from .nnunet_sub.preprocessing.preprocessing import resize_segmentation_ori
    from .nnunet_sub.utilities.one_hot_encoding import to_one_hot
    from .nnunet_sub.training.model_restore import load_model_and_checkpoint_files
    from .nnunet_sub.inference.segmentation_export import save_segmentation_nifti
else:
    from nnunet_sub.preprocessing.preprocessing import resize_segmentation_ori
    from nnunet_sub.utilities.one_hot_encoding import to_one_hot
    from nnunet_sub.training.model_restore import load_model_and_checkpoint_files
    from nnunet_sub.inference.segmentation_export import save_segmentation_nifti

del sys.path[0]

logger = logging.getLogger(__name__)

def get_info_with_sitk_dcm(dcm_path: str):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    spacing = image.GetSpacing()[::-1]
    origin = image.GetOrigin()
    array = sitk.GetArrayFromImage(image)
    shape = array.shape
    physical_shape = np.array(shape) * np.array(spacing)
    direction = image.GetDirection()
    informat = {
     "spacing":spacing,
     "origin":origin,
     "physical_shape":physical_shape,
     "shape":shape,
     "direction":direction
    }
    return array, informat


def get_info_with_sitk_nrrd(nrrd_path: str):
    # 读取NRRD文件
    image = sitk.ReadImage(nrrd_path)
    spacing = image.GetSpacing()[::-1]
    origin = image.GetOrigin()
    direction = image.GetDirection()
    array = sitk.GetArrayFromImage(image)
    shape = array.shape
    physical_shape = np.array(shape) * np.array(spacing)
    informat = {
        "spacing": spacing,
        "origin": origin,
        "physical_shape": physical_shape,
        "shape": shape,
        "direction": direction
    }
    return array, informat

def handle_save_array(path: str, obj: np.ndarray, info: dict):
    writer = sitk.ImageFileWriter()
    writer.UseCompressionOn()
    writer.SetFileName(path)
    img: sitk.Image = sitk.GetImageFromArray(obj)
    img.SetSpacing(info.get('spacing',(1.0, 1.0, 1.0))[::-1])
    img.SetOrigin(info.get('origin',(0.0, 0.0, 0.0)))
    img.SetDirection(info.get('direction',
                              (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))
    writer.Execute(img)

def nnUNet_pred(
    iso_data,
    model_path: str = None,
    folds=None,
    segs_from_prev_stage=None,
    do_tta=True,
    use_gaussian=True,
    mixed_precision=True,
    all_in_gpu=True,
    step_size=0.5,
    checkpoint_name="model_final_checkpoint",
    trt_engine = None,
):
    torch.cuda.empty_cache()

    trainer, params = load_model_and_checkpoint_files(
        model_path,
        folds,
        mixed_precision=mixed_precision,
        checkpoint_name=checkpoint_name,
    )

    trainer.data_aug_params["mirror_axes"] = (2,)
    if trt_engine is None:
        trainer.patch_size = np.array([80,  160,  160])
    else:
        trainer.patch_size = trt_engine.input_shape[2:]
        trainer.network.trt_engine = trt_engine

    d, _, dct = trainer.preprocess_patient(iso_data)
    classes = list(range(1, trainer.num_classes))

    if segs_from_prev_stage is not None:
        seg_prev = segs_from_prev_stage.transpose(trainer.plans["transpose_forward"])
        seg_reshaped = resize_segmentation_ori(seg_prev, d.shape[1:], order=1)
        seg_reshaped = to_one_hot(seg_reshaped, classes)
        d = np.vstack((d, seg_reshaped)).astype(np.float32)

    # preallocate the output arrays
    # same dtype as the return value in predict_preprocessed_data_return_seg_and_softmax (saves time)
    all_softmax_outputs = np.zeros(
        (len(params), trainer.num_classes, *d.shape[1:]), dtype=np.float16
    )
    all_seg_outputs = np.zeros((len(params), *d.shape[1:]), dtype=int)

    for i, p in enumerate(params):
        trainer.load_checkpoint_ram(p, False)
        res = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=do_tta,
            mirror_axes=trainer.data_aug_params["mirror_axes"],
            use_sliding_window=True,
            step_size=step_size,
            use_gaussian=use_gaussian,
            all_in_gpu=all_in_gpu,
            verbose=False,
            mixed_precision=mixed_precision,
        )
        if len(params) > 1:
            all_softmax_outputs[i] = res[1]
        all_seg_outputs[i] = res[0]

    if hasattr(trainer, "regions_class_order"):
        region_class_order = trainer.regions_class_order
    else:
        region_class_order = None
    assert region_class_order is None, (
        "predict_cases_fastest can only work with regular softmax predictions "
        "and is therefore unable to handle trainer classes with region_class_order"
    )

    logger.info(f"aggregating predictions")
    if len(params) > 1:
        softmax_mean = np.mean(all_softmax_outputs, 0)
        seg = softmax_mean.argmax(0)
    else:
        seg = all_seg_outputs[0]

    logger.info(f"applying transpose_backward")
    transpose_forward = trainer.plans.get("transpose_forward")
    if transpose_forward is not None:
        transpose_backward = trainer.plans.get("transpose_backward")
        seg = seg.transpose([i for i in transpose_backward])

    results = save_segmentation_nifti(seg, None, dct, 0, None)
    return results


def main(root_path, file_name, sampling=1) -> NoReturn:
    try:iso_raw, informat = get_info_with_sitk_nrrd(os.path.join(root_path,"input",file_name))
    except:iso_raw, informat = get_info_with_sitk_dcm(os.path.join(root_path, "input", file_name))
    ori_raw = normalize_spacing(iso_raw, np.array(informat['shape'])/sampling)
    fullres3d_model_path = os.path.join(root_path, "model", "3d_fullres_focal")
    try:
        from trt_engine import TRTEngine
        trt_engine = TRTEngine(os.path.join(root_path, "model", 'model_final_checkpoint.engine'))
    except:
        trt_engine = None
    ori_pred = nnUNet_pred(
        ori_raw,
        model_path=fullres3d_model_path,
        folds=0,
        use_gaussian=True,
        do_tta=True,
        step_size=1,
        checkpoint_name="model_final_checkpoint",
        trt_engine=trt_engine,
    )
    iso_pred = normalize_spacing(ori_pred, np.array(informat['shape']))
    iso_pred = resegment_cuspid(iso_pred)
    handle_save_array(os.path.join(root_path, "output", file_name), iso_pred, informat)

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "TestData.nrrd"
    test = main(root_path,file_name)