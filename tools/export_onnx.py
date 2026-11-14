"""Export bobot YOLO (.pt) ke ONNX untuk backend onnxruntime (L10), atau ke TensorRT di Jetson.

Contoh:
  python -m tools.export_onnx models/weights/cabin_best.pt
  python -m tools.export_onnx models/weights/yolo11n.pt --imgsz 640
  # DI Jetson (Fase 2), engine harus di-build di perangkat itu sendiri:
  python -m tools.export_onnx models/weights/cabin_best.pt --format engine --half
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(weights: str | Path, imgsz: int = 640, fmt: str = "onnx", half: bool = False,
           opset: int | None = None) -> Path:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    kw = dict(format=fmt, imgsz=imgsz, half=half, dynamic=False)
    if fmt == "onnx":
        kw.update(simplify=True, nms=False)
        if opset:
            kw["opset"] = opset
    out = Path(m.export(**kw))
    target = Path(weights).with_suffix(out.suffix)
    if out.resolve() != target.resolve():
        shutil.copy2(out, target)
    return target


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine"])
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--opset", type=int, default=None)
    a = ap.parse_args(argv)
    p = export(a.weights, a.imgsz, a.format, a.half, a.opset)
    print(f"Hasil: {p}")


if __name__ == "__main__":
    main()
