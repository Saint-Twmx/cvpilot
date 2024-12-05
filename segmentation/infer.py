# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:31:51 2024

@author: CYC
"""
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np 
from scipy.ndimage import gaussian_filter
import pickle
import torch
import openvino as ov
import torch.nn.functional as F
import os
import gc
import time
class netmodel():
    def __init__(self, PU="CPU", root_path=r""):
        self.root_path = root_path
        assert any([f.endswith('.xml') for f in os.listdir(self.root_path)]), "未找到openvino模型"
        assert any([f.endswith('.bin') for f in os.listdir(self.root_path)]), "未找到openvino模型"
        self.pu = PU
        self.device = None
        self.processing_pkl()
        self.read_openvino()

    def read_openvino(self):
        loaded_model = ov.Core().read_model(os.path.join(self.root_path,"model.xml"))
        core = ov.Core()
        available_devices = core.available_devices
        if self.device is None:
            if self.pu == 'auto' and len(available_devices)>1:
                self.auto_device(loaded_model, available_devices)
            elif self.pu in available_devices:
                self.device = self.pu
            else:
                self.device = available_devices[-1]
            print(f"Using device: {self.device}")
        self.model = ov.compile_model(loaded_model, device_name=self.device)

    def get_info_with_sitk_dcm(self, dcm_path: str):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        self.data = sitk.GetArrayFromImage(image)
        gc.collect()

    def get_array_info(self, array):
        self.data = array


    def crop_to_nonzero_new(self, data, seg=None, nonzero_label=-1):
        nonzero_mask = (data != 0) #没有填充空洞，也不知道填充弄个啥
        bbox = [[0,data.shape[1]],[0,data.shape[2]],[0,data.shape[3]]]
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
        return data, seg, bbox

    def crop_new(self, data, properties, seg=None):
        data, seg, bbox = self.crop_to_nonzero_new(data, seg, nonzero_label=-1)
        properties["crop_bbox"] = bbox
        properties["classes"] = np.array([-1, 0])
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties
    
    def crop_from_file(self, data_file, seg_file=None):
        properties = OrderedDict()
        if isinstance(data_file, str):
            data_itk = sitk.ReadImage(data_file)
            properties["original_size_of_raw_data"] = np.array(data_itk.GetSize())[
                [2, 1, 0]
            ]
            properties["original_spacing"] = np.array(data_itk.GetSpacing())[[2, 1, 0]]
            properties["itk_origin"] = data_itk.GetOrigin()
            properties["itk_spacing"] = data_itk.GetSpacing()
            properties["itk_direction"] = data_itk.GetDirection()
            data_npy = np.vstack([sitk.GetArrayFromImage(data_itk)[None]])
        elif isinstance(data_file, np.ndarray):
            data_itk = data_file
            properties["original_size_of_raw_data"] = data_itk.shape
            properties["original_spacing"] = np.array([1, 1, 1])
            properties["itk_spacing"] = np.array([1, 1, 1])
            data_npy = np.vstack([data_itk[None]])

        properties["seg_file"] = seg_file
        data = data_npy.astype(np.float32)
        if seg_file is not None:
            seg_itk = sitk.ReadImage(seg_file)
            seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
        else:
            seg_npy = None
        seg = seg_npy
        del data_file
        gc.collect()
        return self.crop_new(data, properties, seg)
 
    def resample_and_normalize(self, data, target_spacing, properties,
                               use_nonzero_mask, intensityproperties,
                               seg=None, force_separate_z=None):
        data[np.isnan(data)] = 0
        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        for c in range(len(data)):
            assert (
                intensityproperties is not None
            ), "ERROR: if there is a CT then we need intensity properties"
            mean_intensity = intensityproperties[c]["mean"]
            std_intensity = intensityproperties[c]["sd"]
            lower_bound = intensityproperties[c]["percentile_00_5"]
            upper_bound = intensityproperties[c]["percentile_99_5"]
            data[c] = np.clip(data[c], lower_bound, upper_bound)
            data[c] = (data[c] - mean_intensity) / std_intensity
            if use_nonzero_mask[c]:
                data[c][seg[-1] < 0] = 0
        return data, seg, properties
    

    def preprocess_test_case(self, target_spacing, use_nonzero_mask, intensityproperties,
                             seg_file=None, force_separate_z=None):
        data, seg, properties = self.crop_from_file(self.data, seg_file)
        data = data.transpose((0, *[i + 1 for i in [0, 1, 2]]))
        data, seg, properties = self.resample_and_normalize(data, target_spacing, properties,
                                                            use_nonzero_mask, intensityproperties,)
        return data.astype(np.float32), seg, properties


    def pad_nd_image(self, image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
        if kwargs is None:
            kwargs = {'constant_values': 0}

        if new_shape is not None:
            old_shape = np.array(image.shape[-len(new_shape):])
        else:
            assert shape_must_be_divisible_by is not None
            assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
            new_shape = image.shape[-len(shape_must_be_divisible_by):]
            old_shape = new_shape

        num_axes_nopad = len(image.shape) - len(new_shape)

        new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

        if not isinstance(new_shape, np.ndarray):
            new_shape = np.array(new_shape)

        if shape_must_be_divisible_by is not None:
            if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
                shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
            else:
                assert len(shape_must_be_divisible_by) == len(new_shape)

            for i in range(len(new_shape)):
                if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                    new_shape[i] -= shape_must_be_divisible_by[i]

            new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

        difference = new_shape - old_shape
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

        if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
            res = np.pad(image, pad_list, mode, **kwargs)
        else:
            res = image

        if not return_slicer:
            return res
        else:
            pad_list = np.array(pad_list)
            pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
            slicer = list(slice(*i) for i in pad_list)
            return res, slicer
    

    def _compute_steps_for_sliding_window(
        self, patch_size, image_size, step_size: float):
        assert [
            i >= j for i, j in zip(image_size, patch_size)
        ], "image size must be as large or larger than patch_size"
        assert (
            0 < step_size <= 1
        ), "step_size must be larger than 0 and smaller or equal to 1"

        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [
            int(np.ceil((i - k) / j)) + 1
            for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)
        ]

        steps = []
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = (
                    99999999999
                )

            steps_here = [
                int(np.round(actual_step_size * i)) for i in range(num_steps[dim])
            ]

            steps.append(steps_here)

        return steps

    def _get_gaussian(self, patch_size, sigma_scale=1.0 / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(
            tmp, sigmas, 0, mode="constant", cval=0
        )
        gaussian_importance_map = (
            gaussian_importance_map / np.max(gaussian_importance_map) * 1
        )
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0]
        )
        return gaussian_importance_map
    

    def maybe_to_torch(self, d):
        if isinstance(d, list):
            d = [self.maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
        elif not isinstance(d, torch.Tensor):
            d = torch.from_numpy(d).float()
        return d

    def win_infer(self, data):
        result = self.model(data)
        x = torch.from_numpy(result[self.model.output(0)])
        softmax_helper = lambda x: F.softmax(x, 1)
        return softmax_helper(x)

    def _internal_maybe_mirror_and_pred_3D(self, patch_size, x, mirror_axes, do_mirroring, mult):
        x = self.maybe_to_torch(x)
        result_torch = torch.zeros([1] + patch_size, dtype=torch.float)
        mult = self.maybe_to_torch(mult)

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                tmp = x.clone().numpy()
                pred = self.win_infer(tmp)
                result_torch += 1 / num_results * pred
                del pred

            if m == 1 and (2 in mirror_axes):
                tmp = torch.flip(x, (4,)).clone().numpy()
                pred = self.win_infer(tmp)
                result_torch += 1 / num_results * torch.flip(pred, (4,))
                del pred

            if m == 2 and (1 in mirror_axes):
                tmp = torch.flip(x, (3,)).clone().numpy()
                pred = self.win_infer(tmp)
                result_torch += 1 / num_results * torch.flip(pred, (3,))
                del pred

            gc.collect()

        if mult is not None:
            result_torch[:, :] *= mult
        del x
        del mult
        gc.collect()
        return result_torch

    def processing_pkl(self):
        pkl_path = os.path.join(self.root_path, 'model.pkl')
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        self.num_classes = len(pkl_data["plans"]["all_classes"]) + 1
        self.use_nonzero_mask = pkl_data["plans"]["use_mask_for_norm"]
        self.intensityproperties = pkl_data["plans"]["dataset_properties"]["intensityproperties"]
        self.patch_size = pkl_data["plans"]["plans_per_stage"][1]["patch_size"].tolist()
        del pkl_data
        gc.collect()

    def predict_3D(self):
        d, _, dct = self.preprocess_test_case(np.array([1., 1., 1.]),
                                              self.use_nonzero_mask, self.intensityproperties)

        data, slicer = self.pad_nd_image(
            d, self.patch_size, 'constant', {"constant_values": 0}, True, None
        )
        data_shape = data.shape
        steps = self._compute_steps_for_sliding_window(self.patch_size, data_shape[1:], 1)
        gaussian_importance_map = self._get_gaussian(self.patch_size, sigma_scale=1.0 / 8)
        add_for_nb_of_preds = np.array(gaussian_importance_map.copy())

        aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data_shape[1:]), dtype=np.float32)
        aggregated_results = np.zeros([self.num_classes] + list(data_shape[1:]), dtype=np.float32)
        ALL_NUM = len(steps[0]) * len(steps[1]) * len(steps[2])
        TMP_NUM = 0
        for x in steps[0]:
            lb_x = x
            ub_x = x + self.patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + self.patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + self.patch_size[2]
                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        [self.num_classes] + self.patch_size,
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z],
                        (0,),
                        True,
                        gaussian_importance_map,
                    )[0]

                    predicted_patch = predicted_patch.cpu().numpy()
                    aggregated_results[
                    :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z
                    ] += predicted_patch

                    aggregated_nb_of_predictions[
                    :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z
                    ] += add_for_nb_of_preds
                    del predicted_patch
                    gc.collect()

                    TMP_NUM+=1

        slicer = tuple(
            [
                slice(0, aggregated_results.shape[i])
                for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
            ]
            + slicer[1:]
        )
        aggregated_results = aggregated_results[slicer]
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions
        gc.collect()
        # self.handle_save_array(args, aggregated_results.argmax(0))
        return aggregated_results.argmax(0)

def serial_infer(dcm, PU='CPU', root_path=r""):
    timea = time.time()
    model = netmodel(PU,root_path)
    model.get_array_info(dcm)
    result = model.predict_3D()
    print(f"耗时：{time.time() - timea}")
    return result



    
    