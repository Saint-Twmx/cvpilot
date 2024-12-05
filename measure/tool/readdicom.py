import SimpleITK as sitk
import numpy as np
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
    direction = image.GetDirection()  # 获取方向信息
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
    # 提取图像信息
    spacing = image.GetSpacing()[::-1]
    origin = image.GetOrigin()
    direction = image.GetDirection()  # 获取方向信息
    array = sitk.GetArrayFromImage(image)
    shape = array.shape
    physical_shape = np.array(shape) * np.array(spacing)
    # 创建信息字典
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
    img.SetSpacing(info.get('spacing',(1.0, 1.0, 1.0))[::-1]) # 旋转保存成（XYZ），因为被人读取后也默认旋转。。
    img.SetOrigin(info.get('origin',(0.0, 0.0, 0.0)))
    img.SetDirection(info.get('direction',
                              (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))
    writer.Execute(img)