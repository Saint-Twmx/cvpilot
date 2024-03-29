import time

import onnxruntime
import onnx
import torch
import os

curdir = os.path.abspath(os.path.dirname(__file__))
from nnunet_sub.training.model_restore import (
    load_model_and_checkpoint_files,
)
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import SimpleITK as sitk
from torch.cuda.amp import autocast
from trt_engine import TRTEngine


def prepare_model():
    fullres3d_model_path = "/home/yuchencai/TAVIgator/model/3d_fullres_focal"
    trainer, params = load_model_and_checkpoint_files(
        fullres3d_model_path,
        0,
        mixed_precision=False,
        checkpoint_name="model_final_checkpoint",
    )
    trainer.load_checkpoint_ram(params[0], False)
    trainer.network.do_ds = False
    trainer.network.eval()
    return trainer.network


def generate_onnx(model, infer_sample, onnx_name):

    torch.onnx.export(
        model,
        infer_sample,
        onnx_name,
        export_params=True,
        opset_version=9,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


def test_onnx(onnx_name, infer_sample):

    onnx_model = onnx.load(onnx_name)
    onnx.checker.check_model(onnx_model)
    onnx_session = onnxruntime.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = onnx_session.get_inputs()[0].name
    output_name = [i.name for i in onnx_session.get_outputs()]

    start_time = time.time()
    onnx_result = onnx_session.run(output_name, {input_name: infer_sample})[0]
    print(f"onnx inference time {time.time() - start_time}")

    onnx_pred = np.argmax(onnx_result, axis=1)
    return onnx_result, onnx_pred


def generate_tensorrt(onnx_file_path, engine_file_path, flop=16):
    trt_logger = trt.Logger(trt.Logger.ERROR)  # trt.Logger.ERROR
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, trt_logger)
    # parse ONNX
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("Completed parsing ONNX file")

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ", engine_file_path)

    print("Creating Tensorrt Engine")

    config = builder.create_builder_config()
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print("Serialized Engine Saved at: ", engine_file_path)
    return serialized_engine


def _load_engine(engine_file_path):
    trt_logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_file_path, "rb") as f:
        with trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def test_trt(trt_engine: TRTEngine, data_input):
    # Code from blog.csdn.net/TracelessLe
    result = trt_engine(data_input)
    return result, torch.argmax(result, axis=1)


def test_original(original_model, infer_sample):
    cuda_tensor = torch.from_numpy(infer_sample).cuda()
    with autocast():
        with torch.no_grad():
            time1 = time.time()
            original_result = original_model(cuda_tensor)
            print(f"original inference time {time.time() - time1}")
    original_pred = torch.argmax(original_result, dim=1)
    return original_result, original_pred


if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.cuda.set_device(1)
    device_prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    torch.backends.cudnn.enabled = True
    trt.init_libnvinfer_plugins(None, "")
    torch.random.manual_seed(666)

    patch_size = [1, 1, 80, 160, 160]
    init = [80, 100, 100]
    infer_sample = np.random.randn(*patch_size).astype(np.float32)

    onnx_name = "/home/yuchencai/TAVIgator/model/model_final_checkpoint.onnx"
    # engine_name = "20221117_root"
    # engine_name += f"_SM_{device_prop.major}_{device_prop.minor}.engine"
    engine_name = "/home/yuchencai/TAVIgator/model/model_final_checkpoint.engine"

    model = prepare_model()
    generate_onnx(model, torch.from_numpy(infer_sample).cuda(), onnx_name)
    generate_tensorrt(onnx_name, engine_name, device_prop)

    trt_engine = TRTEngine(engine_name)
    for i in range(20):

        # original_result, original_pred = test_original(model, sub_infer_sample)
        # onnx_result, onnx_pred = test_onnx(onnx_name, sub_infer_sample)

        tensorrt_pred = test_trt(trt_engine, infer_sample.astype(np.float32))
