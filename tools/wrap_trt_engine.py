"""Bungkus engine TensorRT mentah (hasil trtexec) dengan header metadata Ultralytics.

Dipakai di Jetson jika `tools.export_onnx --format engine` gagal (mis. crash saat export ONNX):
  /usr/src/tensorrt/bin/trtexec --onnx=models/weights/cabin_best.onnx \
      --saveEngine=models/weights/cabin_best.trt --fp16
  python -m tools.wrap_trt_engine models/weights/cabin_best.trt models/weights/cabin_best.onnx \
      models/weights/cabin_best.engine

Format .engine Ultralytics: 4 byte panjang metadata (little-endian) + JSON metadata + engine TensorRT.
Metadata (names, imgsz, stride, task, ...) disalin dari metadata_props ONNX yang ditulis Ultralytics.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def onnx_metadata(onnx_path: str | Path) -> dict:
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    meta: dict = {}
    for p in m.metadata_props:
        try:
            meta[p.key] = ast.literal_eval(p.value)
        except (ValueError, SyntaxError):
            meta[p.key] = p.value
    if "names" not in meta:
        raise ValueError(f"{onnx_path}: metadata 'names' tidak ada (bukan ONNX hasil export Ultralytics?)")
    meta["names"] = {int(k): v for k, v in meta["names"].items()}
    return meta


def wrap(trt_path: str | Path, onnx_path: str | Path, out_path: str | Path, half: bool = True) -> Path:
    meta = onnx_metadata(onnx_path)
    meta.setdefault("task", "detect")
    meta.setdefault("batch", 1)
    meta["half"] = half
    raw = Path(trt_path).read_bytes()
    js = json.dumps(meta).encode()
    out = Path(out_path)
    with open(out, "wb") as f:
        f.write(len(js).to_bytes(4, byteorder="little", signed=True))
        f.write(js)
        f.write(raw)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trt", help="engine mentah dari trtexec (.trt)")
    ap.add_argument("onnx", help="ONNX sumber engine (untuk metadata)")
    ap.add_argument("out", help="keluaran .engine untuk Ultralytics")
    ap.add_argument("--fp32", action="store_true", help="engine dibuild tanpa --fp16")
    a = ap.parse_args(argv)
    p = wrap(a.trt, a.onnx, a.out, half=not a.fp32)
    print(f"[wrap] {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
