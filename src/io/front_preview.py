"""Pratinjau + rekaman kamera depan di thread sendiri, opsional dengan deteksi (FrontDetectWorker).

Dipakai di Jetson: kamera depan terlihat di tampilan web (:8090, panel kedua) dan bisa direkam. Dengan
`front_preview.detect.enabled`, YOLO + ByteTrack berjalan di thread lain (maks. `detect.fps`) dan kotak
orang/rintangan digambar di kanal web. Event F-xx belum dikirim (belum disepakati HSE GMF).
Thread terpisah supaya deteksi kabin tidak ikut melambat. Kalau kamera dicabut, dicoba buka lagi tiap `retry_s`.

Hemat CPU: kalau kamera mengirim H.264 (src.capture.uvch264), aliran itu langsung diteruskan ke sink H.264
(rekaman, streaming) tanpa dekode. JPEG hanya didekode + diputar saat benar-benar dipakai: ada penonton web
(dibatasi web.fps) atau perekam MJPG cadangan mengambil frame itu.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(img: np.ndarray, deg: int) -> np.ndarray:
    code = _ROT.get(int(deg) % 360)
    if int(deg) % 360 not in _ROT:
        raise ValueError(f"rotate harus 0/90/180/270, bukan {deg}")
    return img if code is None else cv2.rotate(img, code)


def preview_source_cfg(source_cfg: dict) -> dict:
    """Sumber untuk pratinjau: kalau mode H.264 aktif, JPEG tidak didekode di sumber (dekode saat perlu)."""
    cfg = dict(source_cfg)
    if (cfg.get("h264_mux") or {}).get("enabled", False):
        cfg["h264_mux"] = dict(cfg["h264_mux"], decode=False)
    return cfg


class FrontDetectWorker:
    """Deteksi depan (FrontPipeline: YOLO + ByteTrack) di thread sendiri, maksimal `fps` per detik.

    Thread kamera hanya menaruh frame terbaru di satu slot (frame lama ditimpa), jadi rekaman/streaming
    H.264 tidak pernah menunggu model. Hasil digambar dan dikirim ke kanal web "front".
    """

    def __init__(self, pipe, fps: float, rotate_deg: int = 0, web=None, log_every_s: float = 5.0,
                 decode_reduce: int = 1, dets=None):
        self.pipe = pipe
        self.dets = dets                    # DetsPoster: kotak untuk dashboard (video asli + kotak di browser)
        self.period_s = 1.0 / max(float(fps), 0.1)
        # JPEG didekode di 1/N ukuran (YOLO memakai input 640 -> 720p penuh tidak perlu). Jarak tetap benar:
        # DistanceEstimator menskalakan intrinsik ke ukuran gambar.
        self.decode_reduce = int(decode_reduce)
        self.rotate = rotate_deg
        self.web = web
        self.log_every_s = float(log_every_s)
        self.fps = 0.0
        self.last = None                    # FrontResult terakhir
        self.last_view = None               # (FrontResult, (h, w) gambar tempat hasil itu dihitung)
        # False -> FrontPreview yang menggambar kotak terakhir di setiap frame kamera (video lebih halus
        # daripada laju deteksi); True -> worker mengirim gambar hasil deteksi saja (maks. `fps`)
        self.publish = True
        self.errors = 0
        self._slot = None
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="front-detect", daemon=True)
        self._thread.start()

    def offer(self, fr) -> None:
        with self._cv:
            self._slot = fr
            self._cv.notify()

    def _take(self):
        with self._cv:
            while self._slot is None and not self._stop.is_set():
                self._cv.wait(0.5)
            fr, self._slot = self._slot, None
            return fr

    def _run(self) -> None:
        import time

        from src.capture import Frame
        from src.capture.uvch264 import decode_frame

        n, t_log, t_next = 0, time.monotonic(), 0.0
        while not self._stop.is_set():
            wait = t_next - time.monotonic()
            if wait > 0:
                self._stop.wait(wait)
            fr = self._take()
            if fr is None:
                continue
            t_next = time.monotonic() + self.period_s
            try:
                img = decode_frame(fr, self.decode_reduce)
                if img is None:
                    continue
                img = rotate(img, self.rotate)
                res = self.pipe.process(Frame(img, fr.ts_ms, fr.index, fr.video_t_s))
                self.last = res
                self.last_view = (res, img.shape[:2])   # satu tuple: dibaca thread kamera tanpa kunci
                if self.dets is not None:
                    from src.front.pipeline import dets_message
                    self.dets.push(dets_message(res, img.shape[1], img.shape[0], self.rotate))
                n += 1
                if self.publish and self.web is not None and self.web.wants("front"):
                    self.web.publish(self.pipe.draw(img, res, self.fps), "front")
            except Exception as e:   # model gagal satu frame: lanjut, jangan matikan kamera depan
                self.errors += 1
                if self.errors <= 3 or self.errors % 100 == 0:
                    print(f"[depan] deteksi gagal ({self.errors}x): {e!r}", flush=True)
            dt = time.monotonic() - t_log
            if dt >= self.log_every_s:
                self.fps = n / dt
                n_people = sum(1 for t in (self.last.tracks if self.last else []) if t.cls == "person")
                print(f"[fps-depan] {self.fps:5.1f} FPS | orang {n_people}", flush=True)
                n, t_log = 0, time.monotonic()

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify()
        self._thread.join(timeout_s)
        if self.dets is not None:
            self.dets.close()


def scale_result(res, sx: float, sy: float):
    """Salinan FrontResult dengan kotak diskalakan (deteksi di gambar kecil, gambar tampilan lebih besar)."""
    import dataclasses
    if sx == 1.0 and sy == 1.0:
        return res
    tracks = [dataclasses.replace(t, box=(t.box[0] * sx, t.box[1] * sy, t.box[2] * sx, t.box[3] * sy))
              for t in res.tracks]
    return dataclasses.replace(res, tracks=tracks)


class FrontPreview:
    def __init__(self, source_cfg: dict, cfg: dict, web=None, recorder=None, h264_sinks=None,
                 source_factory=None, detector: FrontDetectWorker | None = None):
        self.source_cfg = preview_source_cfg(source_cfg)
        self.rotate = int(cfg.get("rotate", 0))
        self.retry_s = float(cfg.get("retry_s", 5))
        self.web = web
        self.recorder = recorder            # MJPG (cuplik ~fps); cadangan kalau kamera tidak mengirim H.264
        self.h264_sinks = [s for s in (h264_sinks or []) if s is not None]   # rekaman / streaming H.264
        self.detector = detector            # ada -> kotak deteksi terakhir digambar di kanal web/overlay
        # Video depan mengikuti laju kamera (dibatasi fps tiap keluaran), bukan laju deteksi (~10 fps):
        # dekode di 1/N ukuran (1 = 720p penuh, paling tajam; 2 = 640x360, hemat CPU)
        self.view_reduce = int((cfg.get("view") or {}).get("decode_reduce", 1))
        if detector is not None:
            detector.publish = False
        self.frames = 0
        self.decoded = 0                    # berapa frame yang benar-benar didekode (diagnosis CPU)
        self.connected = False
        self._src = None
        self._mode: str | None = None
        if source_factory is None:
            from src.capture import make_source
            source_factory = make_source
        self._make_source = source_factory
        self._stop = threading.Event()
        if web is not None:
            web.add_channel("front", cfg.get("label", "Kamera depan"))
        self._thread = threading.Thread(target=self._run, name="front-preview", daemon=True)
        self._thread.start()

    def _handle(self, fr) -> None:
        from src.capture.uvch264 import decode_frame
        au = fr.meta.get("h264")
        alive = [s for s in self.h264_sinks if getattr(s, "stopped_reason", None) is None]
        if au is not None:
            for s in alive:
                s.push(au, fr.ts_ms, fr.meta.get("h264_idr"))
        # rekaman H.264 berjalan -> MJPG tidak perlu; kamera tanpa H.264 atau perekam H.264 berhenti -> MJPG
        h264_recording = au is not None and any(getattr(s, "records", False) for s in alive)
        if h264_recording:
            self._log_mode("H.264 dari kamera")
        need_rec = not h264_recording and self.recorder is not None and self.recorder.wants(fr.ts_ms)
        if self.detector is not None:
            self.detector.offer(fr)         # worker mendekode + mendeteksi di thread sendiri
        need_web = self.web is not None and self.web.wants("front")
        if need_rec or need_web:
            img = decode_frame(fr, 1 if need_rec else self.view_reduce)
            if img is None:
                return
            self.decoded += 1
            img = rotate(img, self.rotate)
            if need_rec:
                self._log_mode("MJPG")
                self.recorder.push(img, fr.ts_ms)
            if need_web:
                self.web.publish(self._annotate(img), "front")

    def _annotate(self, img):
        det = self.detector
        view = det.last_view if det is not None else None
        if view is None:
            return img
        res, shape = view
        h, w = img.shape[:2]
        return det.pipe.draw(img, scale_result(res, w / shape[1], h / shape[0]), det.fps)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                src = self._make_source(self.source_cfg)
            except Exception as e:  # kamera tidak ada / sedang dipakai
                if self.connected or self.frames == 0:
                    print(f"[depan] kamera tidak bisa dibuka: {e}; coba lagi tiap {self.retry_s:.0f} s", flush=True)
                self.connected = False
                self._stop.wait(self.retry_s)
                continue
            self.connected = True
            self._src = src
            print("[depan] kamera depan tersambung", flush=True)
            with src:
                for fr in src:
                    if self._stop.is_set():
                        break
                    self._handle(fr)
                    self.frames += 1
            self._src = None
            if not self._stop.is_set():
                print("[depan] kamera depan terputus", flush=True)
                self.connected = False
                self._stop.wait(self.retry_s)

    def _log_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            print(f"[depan] rekaman: {mode}", flush=True)

    def request_idr(self) -> None:
        """Dipanggil sink H.264 saat segmen jatuh tempo: minta keyframe agar segmen cepat berganti."""
        src = self._src
        if src is not None and hasattr(src, "request_idr"):
            src.request_idr()

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout_s)
        if self.detector is not None:
            self.detector.close(timeout_s)
        for rec in [self.recorder, *self.h264_sinks]:
            if rec is not None:
                rec.close()
