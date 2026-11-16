"""Overlay: tampilan ringkas untuk penguji + panel debug (tombol d).

Ringkas:
  kiri atas   status sistem (FPS, kalibrasi, wajah) + keadaan mata/kepala dalam kata-kata
  kanan atas  7 indikator urut prioritas, bar = seberapa dekat ke ambang, berwarna saat aktif
  bawah       banner alert yang sedang terkirim (kuning A-02 / merah A-03) dengan arti kodenya
Debug: semua angka mentah (openness, EAR, yaw/pitch/roll, statistik kamera, info indikator).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.cabin.signals import LEFT_EYE, RIGHT_EYE

# BGR
COL = {"A-02": (0, 200, 255), "A-03": (40, 40, 230), "ok": (80, 190, 60), "txt": (255, 255, 255),
       "dim": (170, 170, 170), "bar": (90, 90, 90), "card": (25, 25, 25), "warn": (0, 170, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# warna + nama kotak YOLO (kelas yang menandakan pelanggaran berwarna merah)
DET_STYLE = {
    "phone": ((255, 0, 255), "ponsel"),
    "helmet": ((80, 190, 60), "helm"),
    "head_no_helmet": ((40, 40, 230), "tanpa helm"),
    "vest": ((80, 190, 60), "rompi"),
    "torso": ((150, 150, 150), "badan"),
}


def _put(img, s, x, y, col=COL["txt"], scale=0.5, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, col, thick, cv2.LINE_AA)


def _card(img, x1, y1, x2, y2, alpha=0.55):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(COL["card"]) * alpha).astype(roi.dtype)


def _dot(img, x, y, col):
    cv2.circle(img, (x, y), 5, col, -1, cv2.LINE_AA)


def _head_words(s, th: dict) -> str:
    d5 = th.get("D05_distraction", {})
    d4 = th.get("D04_head_drop", {})
    if abs(s.roll) > d4.get("roll_deg", 25):
        return "miring"
    if s.yaw > d5.get("yaw_deg", 30):
        return "menoleh"
    if s.yaw < -d5.get("yaw_deg", 30):
        return "menoleh"
    if s.pitch < -d5.get("pitch_down_deg", 25):
        return "menunduk"
    return "lurus"


def _draw_status(img, state, pipeline, fps):
    s = state.sig
    cal = pipeline.calibration
    cal_ok = cal.done
    cal_txt = "OK" if cal_ok else ("berjalan" if cal.active else "populasi")
    _card(img, 6, 6, 250, 78)
    y = 26
    _dot(img, 18, y - 5, COL["ok"] if fps >= 15 else COL["warn"])
    _put(img, f"{fps:4.1f} FPS", 30, y)
    _dot(img, 120, y - 5, COL["ok"] if cal_ok else COL["warn"])
    _put(img, f"kalibrasi {cal_txt}", 132, y)
    y += 24
    if not s.face_present:
        _dot(img, 18, y - 5, COL["A-03"])
        _put(img, "wajah tidak terlihat", 30, y)
    else:
        eye_col = COL["A-03"] if s.eye_closed else COL["ok"]
        _dot(img, 18, y - 5, eye_col)
        _put(img, "mata " + ("TERTUTUP" if s.eye_closed else "terbuka")
             + ("" if s.eye_valid else " (tak terbaca)"), 30, y)
        y += 22
        hw = _head_words(s, pipeline.th)
        _dot(img, 18, y - 5, COL["ok"] if hw == "lurus" else COL["warn"])
        _put(img, f"kepala {hw}", 30, y)
        # lencana menguap (masukan skor kantuk D-01): saat menguap / jumlah menguap di jendela
        d1 = pipeline.indicators.get("D01")
        info = d1.info if d1 is not None else {}
        if not state.blocked_frame:
            if info.get("yawning"):
                _put(img, "MENGUAP", 172, y, COL["warn"], 0.45, 2)
            elif info.get("yawns"):
                _put(img, f"menguap {info['yawns']}x", 172, y, COL["warn"], 0.42)
    if state.blocked_frame:
        _put(img, "KAMERA TERHALANG?", 132, 72, COL["A-03"], 0.45)


def _draw_indicators(img, state, pipeline):
    h, w = img.shape[:2]
    ev = getattr(pipeline.decision, "c", {}) or {}
    order = ev.get("priority", list(pipeline.indicators))
    short = ev.get("short_labels", {})
    levels = ev.get("levels", {})
    open_code = pipeline.decision.open.code if pipeline.decision.open is not None else None
    row_h, cw = 24, 210
    x0, y0 = w - cw - 6, 6
    _card(img, x0, y0, x0 + cw, y0 + 10 + row_h * len(order))
    for i, code in enumerate(order):
        ind = pipeline.indicators.get(code)
        if ind is None:
            continue
        y = y0 + 22 + i * row_h
        active = code in state.active
        lvl_col = COL.get(levels.get(code, "A-02"), COL["A-02"])
        name = short.get(code, code)
        label = f"{code[0]}-{code[1:]} {name}"
        _put(img, label, x0 + 8, y, COL["txt"] if active else COL["dim"], 0.45, 2 if code == open_code else 1)
        # bar progres menuju ambang
        bx1, bx2 = x0 + 140, x0 + cw - 10
        cv2.rectangle(img, (bx1, y - 9), (bx2, y - 1), COL["bar"], -1)
        p = float(np.clip(ind.confidence if ind.confidence is not None else 0.0, 0, 1))
        if active:
            p = 1.0
        if p > 0:
            cv2.rectangle(img, (bx1, y - 9), (bx1 + int((bx2 - bx1) * p), y - 1),
                          lvl_col if active else (200, 200, 200), -1)
        if code == open_code:
            cv2.rectangle(img, (x0 + 2, y - 15), (x0 + cw - 2, y + 6), lvl_col, 1)


_YAWN_KEYS = ("yawns", "score", "yawning")


def _yawn_line(s, pipeline, state) -> str:
    d1 = pipeline.indicators.get("D01")
    tracker = getattr(d1, "yawn", None)
    run = tracker.run_s(state.frame.ts_ms) if tracker is not None else 0.0
    info = d1.info if d1 is not None else {}
    mouth = "-" if s.yawn_open is None else f"{s.yawn_open:.2f}"
    return (f"mulut {mouth}  run {run:.1f}s  menguap {info.get('yawns', 0)}"
            f"  skor D01 {info.get('score', d1.confidence if d1 else 0):.2f}")


def _draw_debug(img, state, pipeline, fps):
    s = state.sig
    lines = [
        f"FPS {fps:5.1f}  face {state.timings_ms.get('face', 0):.0f}ms  yolo {pipeline.yolo_status}",
        f"kalibrasi: {pipeline.calibration.status}",
        f"wajah {'ya' if s.face_present else 'TIDAK'}  eye_valid {s.eye_valid}  mata {'TUTUP' if s.eye_closed else 'buka'}",
        f"openness {s.openness:.2f}  EAR {s.ear:.3f}  blink {s.blink:.2f}",
        f"yaw {s.yaw:+6.1f}  pitch {s.pitch:+6.1f}  roll {s.roll:+6.1f}",
        _yawn_line(s, pipeline, state),
        f"cam std {state.cam_stats['std']:.1f} lap {state.cam_stats['lap_var']:.0f} mean {state.cam_stats['mean']:.0f}"
        f"{'  BLOCKED' if state.blocked_frame else ''}",
    ]
    for code, ind in pipeline.indicators.items():
        mark = "*" if code in state.active else " "
        # info menguap D-01 sudah tampil di baris "mulut" -> baris D01 tidak melebar
        items = {k: v for k, v in ind.info.items() if k not in _YAWN_KEYS}
        lines.append(f"{mark}{code} " + " ".join(f"{k}={v}" for k, v in items.items()))
    _card(img, 0, 84, 470, 92 + 17 * len(lines), 0.6)
    for i, line in enumerate(lines):
        _put(img, line, 8, 102 + i * 17, COL["txt"], 0.42)


def _draw_banner(img, pipeline, open_event):
    h, w = img.shape[:2]
    if open_event is None:
        cv2.rectangle(img, (0, h - 22), (w, h), COL["ok"], -1)
        _put(img, "Normal - tidak ada alert", 10, h - 6, (255, 255, 255), 0.5, 1)
        return
    ev = getattr(pipeline.decision, "c", {}) or {}
    col = COL.get(open_event.level, COL["A-02"])
    what = ev.get("labels", {}).get(open_event.code, "")
    lvl = ev.get("level_labels", {}).get(open_event.level, "")
    code = f"{open_event.code[0]}-{open_event.code[1:]}"
    cv2.rectangle(img, (0, h - 64), (w, h), col, -1)
    fg = (0, 0, 0) if open_event.level == "A-02" else (255, 255, 255)   # kontras di kuning / merah
    _put(img, f"{code}  {what}", 10, h - 36, fg, 0.75, 2)
    _put(img, f"{open_event.level}  {lvl}", 10, h - 10, fg, 0.6, 1)


def draw_cabin(state, pipeline, fps: float, open_event=None, debug: bool = False) -> np.ndarray:
    img = state.image.copy()
    s = state.sig
    if state.obs.present and state.obs.landmarks_px is not None:
        lm = state.obs.landmarks_px
        for idx in (LEFT_EYE, RIGHT_EYE):
            pts = lm[idx].astype(int)
            cv2.polylines(img, [pts], True, COL["A-03"] if s.eye_closed else COL["ok"], 1, cv2.LINE_AA)
    if state.yolo:
        for d in state.yolo.dets:
            col, name = DET_STYLE.get(d.cls_name, ((255, 255, 255), d.cls_name))
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if d.cls_name in ("phone", "head_no_helmet") else 1)
            _put(img, f"{name} {d.conf:.2f}", x1 + 3, max(14, y1 - 5), col, 0.45, 1)
    _draw_status(img, state, pipeline, fps)
    _draw_indicators(img, state, pipeline)
    if debug:
        _draw_debug(img, state, pipeline, fps)
    _draw_banner(img, pipeline, open_event)
    return img
