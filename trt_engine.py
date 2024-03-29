import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
from typing import Union


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class TRTEngine:
    def __init__(self, engine_path: str):
        trt.init_libnvinfer_plugins(None, "")
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            with trt.Runtime(trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_shape = tuple(self.context.get_tensor_shape("input"))
        self.output_shape = tuple(self.context.get_tensor_shape("output"))
        self.d_output = Holder(torch.zeros(tuple(self.output_shape), device="cuda"))
        self.stream = cuda.Stream()

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        start_time = time.time()
        assert x.shape == self.input_shape
        if isinstance(x, np.ndarray):
            data = Holder(torch.from_numpy(x).cuda().ravel())
        elif isinstance(x, torch.Tensor):
            data = Holder(x.cuda().ravel())
        else:
            raise NotImplementedError
        self.context.execute_async_v2(
            bindings=[int(data), int(self.d_output)], stream_handle=self.stream.handle
        )
        self.stream.synchronize()
        print(f"trt inference time {time.time() - start_time}")
        return self.d_output.t.reshape(self.output_shape)
