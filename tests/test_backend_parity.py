"""L10: backend ONNX Runtime harus memberi hasil sama dengan PyTorch/Ultralytics untuk model yang sama.

Dua tingkat:
1. ORT murni vs Ultralytics memuat .onnx yang sama -> harus identik (input 640x640 sama).
2. ORT .onnx vs PyTorch .pt -> deteksi kuat harus sama (kelas, IoU > 0,85). Selisih confidence kecil wajar:
   PyTorch memakai letterbox persegi panjang (mis. 480x640), ONNX statis 640x640.
"""
import cv2
import pytest

from src.config import REPO_ROOT

W = REPO_ROOT / "models" / "weights"
IMGS = [REPO_ROOT / "data" / "samples" / n for n in ("bus.jpg", "zidane.jpg")]
PAIRS = [("yolo11n.pt", "yolo11n.onnx"), ("cabin_best.pt", "cabin_best.onnx")]


def _iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def _need(*files):
    for f in files:
        if not (W / f).exists():
            pytest.skip(f"{f} belum ada")
    imgs = [cv2.imread(str(p)) for p in IMGS if p.exists()]
    if not imgs:
        pytest.skip("gambar contoh tidak ada di data/samples")
    return imgs


@pytest.mark.parametrize("pt,onnx", PAIRS)
def test_ort_identical_to_ultralytics_onnx(pt, onnx):
    imgs = _need(onnx)
    from src.backends.ort_backend import OrtYoloDetector
    from src.backends.ultralytics_backend import UltralyticsDetector

    ort = OrtYoloDetector(W / onnx)
    ref = UltralyticsDetector(W / onnx, imgsz=ort.imgsz)   # ukuran input dari ONNX statis
    assert ort.names == ref.names
    # Dibandingkan pada ambang yang dipakai runtime (>= 0,35). Di bawahnya, kotak yang IoU-nya tepat di
    # batas NMS (mis. 0,695 vs iou 0,7) bisa berbeda karena pembulatan implementasi NMS.
    for img in imgs:
        a = [d for d in ref.predict(img, conf=0.2) if d.conf >= 0.35]
        b = [d for d in ort.predict(img, conf=0.2) if d.conf >= 0.35]
        assert abs(len(a) - len(b)) <= 1
        for x in a:
            y = max((d for d in b if d.cls_id == x.cls_id), key=lambda d: _iou(d.xyxy, x.xyxy), default=None)
            assert y is not None, f"{x.cls_name} hilang di ORT"
            if _iou(y.xyxy, x.xyxy) > 0.99:
                assert abs(x.conf - y.conf) < 1e-3
            else:   # hanya boleh untuk duplikat di batas NMS
                assert x.conf < 0.5


@pytest.mark.parametrize("pt,onnx", PAIRS)
def test_ort_matches_pytorch_strong_detections(pt, onnx):
    imgs = _need(pt, onnx)
    from src.backends.ort_backend import OrtYoloDetector
    from src.backends.ultralytics_backend import UltralyticsDetector

    ort = OrtYoloDetector(W / onnx)
    ref = UltralyticsDetector(W / pt, imgsz=ort.imgsz)
    for img in imgs:
        b = ort.predict(img, conf=0.2)
        # PyTorch memakai letterbox persegi panjang, ONNX persegi -> deteksi di batas (~0,5–0,6) bisa
        # muncul/hilang. Bandingkan deteksi yang jelas kuat.
        for d in [d for d in ref.predict(img, conf=0.7)]:
            m = max((x for x in b if x.cls_id == d.cls_id), key=lambda x: _iou(x.xyxy, d.xyxy), default=None)
            assert m is not None, f"{d.cls_name} hilang di ORT"
            assert _iou(m.xyxy, d.xyxy) > 0.85
            h, w = img.shape[:2]
            at_edge = d.xyxy[0] < 2 or d.xyxy[1] < 2 or d.xyxy[2] > w - 2 or d.xyxy[3] > h - 2
            if not at_edge:   # objek terpotong di tepi sensitif terhadap padding letterbox
                assert abs(m.conf - d.conf) < 0.15
