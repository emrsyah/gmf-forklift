"""YOLO di TensorRT murni (Jetson): engine + cuda-python, tanpa torch/ultralytics.

Alasan: di Jetson uji (JetPack 6.2), torch sistem crash (bus error) pada reduksi CUDA yang dipakai
NMS Ultralytics. Backend ini hanya memakai TensorRT untuk inferensi; letterbox, decode, dan NMS
memakai fungsi numpy yang sama dengan backend ONNX Runtime (hasil identik secara logika).

Engine boleh mentah (trtexec --saveEngine) atau berheader metadata Ultralytics (.engine dari
tools.export_onnx / tools.wrap_trt_engine); header dilewati dan dipakai untuk nama kelas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import Detection, Detector
from .ort_backend import postprocess, preprocess


def _read_engine(path: str | Path) -> tuple[bytes, dict]:
    data = Path(path).read_bytes()
    n = int.from_bytes(data[:4], byteorder="little", signed=True)
    if 0 < n < 1_000_000 and data[4:5] == b"{":
        try:
            meta = json.loads(data[4:4 + n].decode("utf-8"))
            return data[4 + n:], meta
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return data, {}


def _check(res):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return res[1] if isinstance(res, tuple) and len(res) == 2 else (res[1:] if isinstance(res, tuple) else None)


class TrtYoloDetector(Detector):
    def __init__(self, weights: str | Path, imgsz: int = 640, names: list[str] | None = None,
                 iou: float = 0.7, max_det: int = 300):
        import tensorrt as trt
        from cuda import cudart

        self._cudart = cudart
        blob, meta = _read_engine(weights)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"Engine TensorRT gagal dimuat: {weights} (build ulang di perangkat ini)")
        self.ctx = self.engine.create_execution_context()
        self.stream = _check(cudart.cudaStreamCreate())

        self._host: dict[str, np.ndarray] = {}
        self._dev: dict[str, int] = {}
        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if any(s < 0 for s in shape):
                raise RuntimeError(f"Engine dinamis tidak didukung ({name}: {shape}); build dengan ukuran tetap")
            host = np.empty(shape, dtype=dtype)
            dev = _check(cudart.cudaMalloc(host.nbytes))
            self._host[name], self._dev[name] = host, dev
            self.ctx.set_tensor_address(name, dev)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        in_shape = self._host[self.input_name].shape
        self.imgsz = int(in_shape[2]) if len(in_shape) == 4 else imgsz
        self.iou = iou
        self.max_det = max_det
        model_names = None
        if isinstance(meta.get("names"), dict):
            d = {int(k): v for k, v in meta["names"].items()}
            model_names = [d[i] for i in sorted(d)]
        self.names = list(names) if names else (model_names or [])

    def _infer(self, x: np.ndarray) -> np.ndarray:
        cu = self._cudart
        hin = self._host[self.input_name]
        np.copyto(hin, x.astype(hin.dtype, copy=False))
        kind_h2d = cu.cudaMemcpyKind.cudaMemcpyHostToDevice
        kind_d2h = cu.cudaMemcpyKind.cudaMemcpyDeviceToHost
        _check(cu.cudaMemcpyAsync(self._dev[self.input_name], hin.ctypes.data, hin.nbytes, kind_h2d, self.stream))
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execute_async_v3 gagal")
        hout = self._host[self.output_name]
        _check(cu.cudaMemcpyAsync(hout.ctypes.data, self._dev[self.output_name], hout.nbytes, kind_d2h, self.stream))
        _check(cu.cudaStreamSynchronize(self.stream))
        return hout

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25) -> list[Detection]:
        x, r, pad = preprocess(image_bgr, self.imgsz)
        out = self._infer(x)
        return postprocess(out, image_bgr.shape, r, pad, conf, self.iou, self.max_det, self.names)

    def __del__(self):
        cu = getattr(self, "_cudart", None)
        if cu is None:
            return
        for d in getattr(self, "_dev", {}).values():
            try:
                cu.cudaFree(d)
            except Exception:
                pass
