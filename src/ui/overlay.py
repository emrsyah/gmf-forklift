"""Overlay debug: sinyal wajah, indikator aktif, level alert, FPS."""
from __future__ import annotations

import cv2
import numpy as np

from src.cabin.signals import LEFT_EYE, RIGHT_EYE

COL = {"A-02": (0, 200, 255), "A-03": (0, 0, 255), "ok": (0, 200, 0), "txt": (255, 255, 255)}


def _txt(img, s, x, y, col=COL["txt"], scale=0.5):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def draw_cabin(state, pipeline, fps: float, open_event=None) -> np.ndarray:
    img = state.image.copy()
    h, w = img.shape[:2]
    s = state.sig
    if state.obs.present and state.obs.landmarks_px is not None:
        lm = state.obs.landmarks_px
        for idx in (LEFT_EYE, RIGHT_EYE):
            pts = lm[idx].astype(int)
            cv2.polylines(img, [pts], True, (0, 0, 255) if s.eye_closed else (0, 255, 0), 1)
        nose = lm[1].astype(int)
        cv2.circle(img, tuple(nose), 3, (255, 255, 0), -1)
    if state.yolo:
        for d in state.yolo.dets:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            _txt(img, f"{d.cls_name} {d.conf:.2f}", x1, max(12, y1 - 4), (255, 0, 255), 0.45)

    # panel kiri
    panel = [
        f"FPS {fps:5.1f}  face {state.timings_ms.get('face', 0):.0f}ms  yolo {pipeline.yolo_status}",
        f"kalibrasi: {pipeline.calibration.status}",
        f"wajah {'ya' if s.face_present else 'TIDAK'}  eye_valid {s.eye_valid}  mata {'TUTUP' if s.eye_closed else 'buka'}",
        f"openness {s.openness:.2f}  EAR {s.ear:.3f}  blink {s.blink:.2f}",
        f"yaw {s.yaw:+6.1f}  pitch {s.pitch:+6.1f}  roll {s.roll:+6.1f}",
        f"cam std {state.cam_stats['std']:.1f} lap {state.cam_stats['lap_var']:.0f} mean {state.cam_stats['mean']:.0f}"
        f"{'  BLOCKED' if state.blocked_frame else ''}",
    ]
    for code, ind in pipeline.indicators.items():
        mark = "*" if code in state.active else " "
        info = " ".join(f"{k}={v}" for k, v in ind.info.items())
        panel.append(f"{mark}{code} {info}")
    for i, line in enumerate(panel):
        _txt(img, line, 8, 18 + i * 17)

    # banner level
    if open_event is not None:
        col = COL.get(open_event.level, COL["A-02"])
        ev_cfg = getattr(pipeline.decision, "c", {}) or {}
        what = ev_cfg.get("labels", {}).get(open_event.code, "")
        lvl = ev_cfg.get("level_labels", {}).get(open_event.level, "")
        cv2.rectangle(img, (0, h - 64), (w, h), col, -1)
        code = open_event.code.replace("D", "D-", 1) if open_event.code.startswith("D") else open_event.code
        fg = (0, 0, 0) if open_event.level == "A-02" else (255, 255, 255)   # kontras di kuning / merah
        cv2.putText(img, f"{code}  {what}", (10, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, fg, 2, cv2.LINE_AA)
        cv2.putText(img, f"{open_event.level}  {lvl}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1,
                    cv2.LINE_AA)
    else:
        cv2.rectangle(img, (0, h - 6), (w, h), COL["ok"], -1)
    return img
