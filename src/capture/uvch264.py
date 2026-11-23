"""H.264 langsung dari kamera UVC (Logitech C930e) lewat extension unit "UVC H.264" (UVCX).

C930e hanya menawarkan YUYV dan MJPG, tetapi lewat XU `a29e7641-…` kamera bisa diminta menyisipkan
aliran H.264 di dalam setiap frame MJPG (segmen APP4). Satu stream memberi dua hal:
- JPEG -> didekode untuk model/tampilan (seperti sebelumnya),
- H.264 -> langsung ke rekaman/streaming tanpa encode di Jetson (Orin Nano tidak punya encoder hardware).

Diuji 2026-11-23 di Jetson: mux harus diset sesudah device dibuka tetapi SEBELUM frame pertama dibaca
(saat streaming, SET_CUR ditolak), jadi setiap kali kamera dibuka ulang konfigurasi harus diulang.
Bagian parser murni (tanpa import Linux) supaya bisa dites di laptop Windows.
"""
from __future__ import annotations

import os
import struct
import sys
import time
import uuid
from typing import Optional

import cv2
import numpy as np

from src.capture import Frame, WebcamSource

_BAD = object()   # penanda frame rusak (read() melewatinya)

UVCX_GUID = uuid.UUID("a29e7641-de04-47e3-8b2b-f4341aff003b")

# UVCX selector
_PROBE, _COMMIT, _PICTURE_TYPE = 0x01, 0x02, 0x09
# UVC request
_SET_CUR, _GET_CUR, _GET_DEF = 0x01, 0x81, 0x87

# uvcx_video_config_probe_commit_t (46 byte, little-endian, packed)
_CFG_FMT = "<IIHHHHHHHHHHBBBBBBBBBBBBBBBBH"
_CFG_NAMES = ("dwFrameInterval dwBitRate bmHints wConfigurationIndex wWidth wHeight wSliceUnits wSliceMode "
              "wProfile wIFramePeriod wEstimatedVideoDelay wEstimatedMaxConfigDelay bUsageType bRateControlMode "
              "bTemporalScaleMode bSpatialScaleMode bSNRScaleMode bStreamMuxOption bStreamFormat bEntropyCABAC "
              "bTimestamp bNumOfReorderFrames bPreviewFlipped bView bReserved1 bReserved2 bStreamID "
              "bSpatialLayerRatio wLeakyBucketSize").split()
CFG_SIZE = struct.calcsize(_CFG_FMT)
assert CFG_SIZE == 46

MUX_H264 = 0b11          # bit0 = mux aktif, bit1 = sisipkan H.264
PROFILES = {"constrained_baseline": 0x4240, "baseline": 0x4200, "main": 0x4D00, "high": 0x6400}


def pack_config(cfg: dict) -> bytes:
    return struct.pack(_CFG_FMT, *(int(cfg[n]) for n in _CFG_NAMES))


def unpack_config(raw: bytes) -> dict:
    return dict(zip(_CFG_NAMES, struct.unpack(_CFG_FMT, raw)))


# ------------------------------------------------------------------------------------------- parser murni
def extract_h264(jpeg: bytes) -> Optional[bytes]:
    """Ambil access unit H.264 (Annex B) dari segmen APP4 sebuah frame MJPG; None jika tidak ada.

    Segmen APP4 pertama berisi header (bcdVersion, wHeaderLen, fourcc 'H264', w, h, interval, delay, PTS)
    lalu u32 ukuran payload; payload bisa berlanjut di segmen APP4 berikutnya (maks ~64 KB per segmen).
    """
    if len(jpeg) < 4 or jpeg[0] != 0xFF or jpeg[1] != 0xD8:
        return None
    j, n = 2, len(jpeg)
    app4 = bytearray()
    while j + 4 <= n and jpeg[j] == 0xFF:
        marker = jpeg[j + 1]
        if marker == 0xDA:          # start of scan: semua segmen APP sudah lewat
            break
        seglen = struct.unpack_from(">H", jpeg, j + 2)[0]
        if marker == 0xE4:
            app4 += jpeg[j + 4:j + 2 + seglen]
        j += 2 + seglen
    if len(app4) < 8:
        return None
    hdr_len = struct.unpack_from("<H", app4, 2)[0]
    if app4[4:8] != b"H264" or len(app4) < hdr_len + 4:
        return None
    size = struct.unpack_from("<I", app4, hdr_len)[0]
    body = bytes(app4[hdr_len + 4:hdr_len + 4 + size])
    return body or None


def nal_units(au: bytes) -> list[tuple[int, int, int]]:
    """(tipe NAL, awal termasuk start code, akhir) untuk setiap NAL di access unit Annex B."""
    starts = []
    i, n = 0, len(au)
    while True:
        k = au.find(b"\x00\x00\x01", i)
        if k < 0:
            break
        sc = k - 1 if k > 0 and au[k - 1] == 0 else k      # start code 4 byte
        if k + 3 < n:
            starts.append((sc, k + 3))
        i = k + 3
    out = []
    for idx, (sc, hdr) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else n
        out.append((au[hdr] & 0x1F, sc, end))
    return out


def nal_types(au: bytes) -> set[int]:
    return {t for t, _, _ in nal_units(au)}


def is_idr(au: bytes) -> bool:
    return 5 in nal_types(au)


def find_xu_unit(descriptors: bytes, guid: uuid.UUID = UVCX_GUID) -> Optional[int]:
    """bUnitID extension unit dengan GUID tertentu dari deskriptor USB mentah (sysfs `descriptors`)."""
    i, n = 0, len(descriptors)
    in_video_control = False
    while i + 2 <= n:
        blen, btype = descriptors[i], descriptors[i + 1]
        if blen < 2:
            break
        if btype == 0x04 and i + 7 <= n:            # interface: kelas 0x0E video, subkelas 0x01 control
            in_video_control = descriptors[i + 5] == 0x0E and descriptors[i + 6] == 0x01
        elif (btype == 0x24 and in_video_control and blen >= 20 and i + 20 <= n
              and descriptors[i + 2] == 0x06):      # CS_INTERFACE / VC_EXTENSION_UNIT
            if uuid.UUID(bytes_le=bytes(descriptors[i + 4:i + 20])) == guid:
                return descriptors[i + 3]
        i += blen
    return None


_REDUCED = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
            8: cv2.IMREAD_REDUCED_COLOR_8}


def decode_jpeg(jpeg: bytes | None, reduce: int = 1) -> Optional[np.ndarray]:
    """reduce 2/4/8: libjpeg mendekode langsung di 1/2, 1/4, 1/8 ukuran (DCT scaling) -> beberapa kali lebih murah
    daripada dekode penuh lalu resize."""
    if not jpeg:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), _REDUCED.get(int(reduce), cv2.IMREAD_COLOR))


def decode_frame(frame: Frame, reduce: int = 1) -> Optional[np.ndarray]:
    """Gambar BGR sebuah frame: `image` kalau sudah ada, kalau tidak dekode `meta["jpeg"]` (decode: false)."""
    if frame.image is not None:
        return frame.image
    return decode_jpeg(frame.meta.get("jpeg"), reduce)


# ------------------------------------------------------------------------------------------- kontrol (Linux)
class UvcxH264:
    """Atur mode H.264 kamera lewat ioctl UVCIOC_CTRL_QUERY. fd dibiarkan terbuka selama kamera dipakai."""

    def __init__(self, dev_path: str, unit_id: int | None = None):
        if not sys.platform.startswith("linux"):
            raise RuntimeError("UVCX H.264 hanya didukung di Linux (uvcvideo)")
        import ctypes
        import fcntl

        class _XuQuery(ctypes.Structure):
            _fields_ = [("unit", ctypes.c_uint8), ("selector", ctypes.c_uint8), ("query", ctypes.c_uint8),
                        ("size", ctypes.c_uint16), ("data", ctypes.c_void_p)]

        self._ctypes, self._fcntl, self._Q = ctypes, fcntl, _XuQuery
        self._ioc = (3 << 30) | (ctypes.sizeof(_XuQuery) << 16) | (ord("u") << 8) | 0x21
        if unit_id is None:
            unit_id = self._unit_from_sysfs(dev_path)
        if unit_id is None:
            raise RuntimeError(f"{dev_path}: kamera tidak punya extension unit UVC H.264")
        self.unit = int(unit_id)
        self.fd = os.open(dev_path, os.O_RDWR)

    @staticmethod
    def _unit_from_sysfs(dev_path: str) -> Optional[int]:
        name = os.path.basename(os.path.realpath(dev_path))
        p = f"/sys/class/video4linux/{name}/device/../descriptors"
        try:
            with open(p, "rb") as f:
                return find_xu_unit(f.read())
        except OSError:
            return None

    def _xu(self, selector: int, query: int, size: int, payload: bytes | None = None) -> bytes:
        c = self._ctypes
        buf = c.create_string_buffer(payload if payload is not None else b"\0" * size, size)
        q = self._Q(self.unit, selector, query, size, c.cast(buf, c.c_void_p))
        self._fcntl.ioctl(self.fd, self._ioc, q)
        return buf.raw

    def configure(self, width: int, height: int, fps: float, bitrate: int, iframe_period_ms: int,
                  profile: int, mux: int = MUX_H264) -> dict:
        """Probe + commit. Kembalikan konfigurasi hasil negosiasi kamera."""
        cfg = unpack_config(self._xu(_PROBE, _GET_DEF, CFG_SIZE))
        cfg.update(dwFrameInterval=int(round(1e7 / fps)), dwBitRate=int(bitrate), bmHints=0,
                   wWidth=int(width), wHeight=int(height), wIFramePeriod=int(iframe_period_ms),
                   wProfile=int(profile), bUsageType=1, bStreamFormat=0, bEntropyCABAC=0,
                   bTimestamp=0, bNumOfReorderFrames=0, bStreamMuxOption=int(mux))
        self._xu(_PROBE, _SET_CUR, CFG_SIZE, pack_config(cfg))
        got = self._xu(_PROBE, _GET_CUR, CFG_SIZE)
        self._xu(_COMMIT, _SET_CUR, CFG_SIZE, got)
        return unpack_config(got)

    def request_idr(self) -> bool:
        """Minta keyframe IDR + SPS/PPS sekarang (boleh gagal: tidak semua firmware mendukung)."""
        try:
            self._xu(_PICTURE_TYPE, _SET_CUR, 4, struct.pack("<HH", 0, 2))
            return True
        except OSError:
            return False

    def disable(self) -> None:
        """Matikan mux (hanya bisa saat kamera tidak streaming)."""
        cfg = unpack_config(self._xu(_PROBE, _GET_CUR, CFG_SIZE))
        cfg["bStreamMuxOption"] = 0
        self._xu(_PROBE, _SET_CUR, CFG_SIZE, pack_config(cfg))
        self._xu(_COMMIT, _SET_CUR, CFG_SIZE, self._xu(_PROBE, _GET_CUR, CFG_SIZE))

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


# ------------------------------------------------------------------------------------------- sumber frame
class UvcH264WebcamSource(WebcamSource):
    """Webcam MJPG + H.264 tersisip. `Frame.image` tetap BGR hasil dekode (interface sama);
    `Frame.meta["h264"]` berisi access unit H.264 kalau ada, `meta["h264_idr"]` True untuk keyframe.

    Kalau kamera/OS tidak mendukung, jalan sebagai webcam biasa (tanpa meta h264) dan mencatat sebabnya.

    `h264.decode: false` (dipakai pratinjau kamera depan): JPEG TIDAK didekode di sini. `Frame.image` = None
    dan `meta["jpeg"]` berisi byte JPEG; pemakai mendekode sendiri hanya saat perlu (lihat `decode_frame`).
    Jangan pakai untuk pipeline deteksi, yang butuh gambar di setiap frame.
    """

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None,
                 fps: float | None = None, api: str = "auto", fourcc: str | None = "MJPG",
                 autofocus: bool | None = None, focus: float | None = None, h264: dict | None = None):
        super().__init__(index, width, height, fps, api, fourcc, autofocus, focus)
        h264 = h264 or {}
        self.decode = bool(h264.get("decode", True))
        self.ctl: UvcxH264 | None = None
        self.negotiated: dict | None = None
        self.h264_error: str | None = None
        self._raw = False
        try:
            # urutan penting (uji Jetson): buka -> mode mentah -> set mux -> baru baca frame pertama
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            self._raw = True
            self.ctl = UvcxH264(f"/dev/video{int(index)}", h264.get("unit_id"))
            w = int(width or self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            hh = int(height or self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            prof = h264.get("profile", "constrained_baseline")
            self.negotiated = self.ctl.configure(
                w, hh, float(fps or self.fps_hint), int(h264.get("bitrate", 3_000_000)),
                int(h264.get("iframe_period_ms", 1000)), PROFILES.get(prof, prof) if isinstance(prof, str)
                else int(prof))
            if self.negotiated["bStreamMuxOption"] != MUX_H264:
                raise RuntimeError(f"kamera menolak mux H.264 (bStreamMuxOption={self.negotiated['bStreamMuxOption']})")
        except Exception as e:  # noqa: BLE001 - fallback ke MJPG biasa, deteksi tetap jalan
            self.h264_error = str(e)
            if self.ctl is not None:
                self.ctl.close()
                self.ctl = None
            print(f"[h264] mode H.264 kamera tidak aktif: {e}; lanjut MJPG biasa", flush=True)

    def read(self) -> Optional[Frame]:
        if not self._raw:
            return super().read()
        for _ in range(int(getattr(self, "max_bad_frames", 30))):   # frame rusak beruntun: loop, bukan rekursi
            fr = self._read_once()
            if fr is not _BAD:
                return fr
        print("[h264] terlalu banyak frame rusak beruntun", flush=True)
        return None

    def _read_once(self):
        ok, raw = self.cap.read()
        if not ok or raw is None:
            return None
        dt = time.monotonic() - self._t0_mono
        meta: dict = {}
        if raw.ndim == 2 and raw.shape[0] == 1:           # buffer MJPG mentah
            jpeg = raw.reshape(-1).tobytes()              # salinan: buffer OpenCV dipakai ulang frame berikutnya
            if self.decode:
                img = decode_jpeg(jpeg)
                if img is None:
                    return _BAD                           # frame rusak dari USB: lewati
            else:
                img = None
                meta["jpeg"] = jpeg
            if self.ctl is not None:
                au = extract_h264(jpeg)
                if au is not None:
                    meta["h264"] = au
                    meta["h264_idr"] = is_idr(au)
        else:                                             # backend sudah mendekode (bukan V4L2)
            img = raw
        f = Frame(img, self._t0_epoch_ms + int(dt * 1000), self._i, dt, meta)
        self._i += 1
        return f

    def request_idr(self) -> bool:
        return self.ctl.request_idr() if self.ctl is not None else False

    def release(self) -> None:
        super().release()
        if self.ctl is not None:
            try:
                self.ctl.disable()      # kembalikan kamera ke MJPG biasa
            except OSError:
                pass
            self.ctl.close()
            self.ctl = None

