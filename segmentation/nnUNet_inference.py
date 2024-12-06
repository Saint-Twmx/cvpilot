from measure.tool.readdicom import get_info_with_sitk_nrrd, handle_save_array
import logging
import os
import sys
from typing import NoReturn
import numpy as np
import torch
import SimpleITK as sitk
from segmentation.infer import serial_infer


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


def main(root_path, file_name, sampling=1) -> NoReturn:
    try:
        array, informat = get_info_with_sitk_nrrd(os.path.join(root_path, "input", file_name))
    except:
        array, informat = get_info_with_sitk_dcm(os.path.join(root_path, "input", file_name))

    model_path = os.path.join(root_path, "model")

    pred = serial_infer(array,"CPU", model_path)

    handle_save_array(os.path.join(root_path, "output", file_name), pred, informat)
