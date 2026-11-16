"""Indikator D-01..D-10 (Tech Spec bagian 4): logika berbasis waktu di atas sinyal per frame.

Semua waktu memakai timestamp SUMBER (`ts_ms`), sehingga replay = live.
Setiap indikator: `update(x: IndicatorInput) -> bool` dan atribut `info` (untuk overlay/log),
`condition_start_ms` (awal kondisi, untuk ts_condition_start_ms), `confidence`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .signals import FaceSignals


@dataclass
class IndicatorInput:
    ts_ms: int
    sig: FaceSignals
    phone_conf: float = 0.0       # skor `phone` tertinggi pada frame (YOLO kabin)
    no_helmet: bool = False       # ada head_no_helmet >= conf
    no_vest: bool = False         # ada torso tanpa vest
    yolo_valid: bool = False      # hasil YOLO tersedia untuk frame ini (atau dibawa dari frame sebelumnya)
    blocked_frame: bool = False   # frame seragam/gelap/terang (D-09, dari frame mentah)
    moving: bool = True           # sinyal kendaraan bergerak (require_moving)


class Indicator:
    code = "D00"

    def __init__(self, cfg: dict):
        self.c = cfg
        self.active = False
        self.condition_start_ms: int | None = None
        self.confidence = 0.0
        self.info: dict = {}
        self._prev_ts: int | None = None

    def _dt(self, ts_ms: int) -> float:
        dt = 0.0 if self._prev_ts is None else max(0.0, (ts_ms - self._prev_ts) / 1000.0)
        self._prev_ts = ts_ms
        return dt

    def update(self, x: IndicatorInput) -> bool:
        raise NotImplementedError

    def reset(self):
        self.__init__(self.c)


class _TimeWindow:
    """Deque (t_ms, dt_s, value) dengan bobot waktu."""

    def __init__(self, window_s: float):
        self.window_ms = window_s * 1000
        self.q: deque = deque()

    def add(self, t_ms: int, dt_s: float, value):
        self.q.append((t_ms, dt_s, value))
        while self.q and self.q[0][0] < t_ms - self.window_ms:
            self.q.popleft()

    def span_s(self) -> float:
        return sum(d for _, d, _ in self.q)

    def weighted_mean(self, fn) -> float:
        tot = sum(d for _, d, _ in self.q)
        if tot <= 0:
            return 0.0
        return sum(d for _, d, v in self.q if fn(v)) / tot


# ---------------------------------------------------------------- D-02 pejaman panjang (A-03)
class D02LongClosure(Indicator):
    code = "D02"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.closed_run = 0.0
        self.gap = 0.0
        self._run_start: int | None = None

    def update(self, x):
        dt = self._dt(x.ts_ms)
        s = x.sig
        if s.face_present and s.eye_valid and s.eye_closed:
            if self._run_start is None:
                self._run_start = x.ts_ms
            self.gap = 0.0
        elif s.face_present and s.eye_valid and not s.eye_closed:
            self._run_start = None
            self.gap = 0.0
        else:
            # wajah hilang / mata tidak terbaca: tahan run selama celah < toleransi
            self.gap += dt
            if self.gap >= self.c.get("face_gap_tolerance_s", 0.3):
                self._run_start = None
        self.closed_run = 0.0 if self._run_start is None else (x.ts_ms - self._run_start) / 1000.0
        self.active = self.closed_run >= self.c.get("closed_s", 1.5)
        self.condition_start_ms = self._run_start
        self.confidence = min(1.0, self.closed_run / max(1e-6, self.c.get("closed_s", 1.5)))
        self.info = {"closed_run_s": round(self.closed_run, 2)}
        return self.active


# ---------------------------------------------------------------- menguap (masukan D-01)
class _YawnTracker:
    """Deteksi menguap dari `sig.yawn_open` (0..1), berbasis timestamp sumber.

    Satu run = mulut terbuka >= yawn_open_min, bertahan selama >= yawn_open_exit (hysteresis).
    Run tidak putus oleh celah singkat: data hilang (wajah hilang / None, mis. tangan menutup mulut)
    <= yawn_gap_s, atau bukaan turun di bawah exit <= yawn_dip_s. Saat run selesai, dihitung sebagai
    menguap jika durasinya dalam [yawn_min_s, yawn_max_s] (lebih pendek = bicara/tertawa,
    lebih panjang = makan/mulut menganga lama, bukan menguap).
    """

    def __init__(self, cfg: dict):
        self.c = cfg
        self.yawns: deque = deque()          # (t_counted_ms, t_start_ms) menguap selesai
        self._start: int | None = None
        self._last_open: int | None = None
        self.value: float | None = None

    @property
    def in_run(self) -> bool:
        return self._start is not None

    def run_s(self, ts_ms: int) -> float:
        return 0.0 if self._start is None else (ts_ms - self._start) / 1000.0

    def _finish(self, ts_ms: int):
        dur = (self._last_open - self._start) / 1000.0
        if self.c.get("yawn_min_s", 2.0) <= dur <= self.c.get("yawn_max_s", 10.0):
            self.yawns.append((ts_ms, self._start))
        self._start = self._last_open = None

    def update(self, ts_ms: int, s: FaceSignals):
        c = self.c
        v = s.yawn_open if s.face_present else None
        self.value = v
        thr = c.get("yawn_open_exit", 0.40) if self.in_run else c.get("yawn_open_min", 0.55)
        if v is not None and v >= thr:
            if self._start is None:
                self._start = ts_ms
            self._last_open = ts_ms
        elif self.in_run:
            tol = c.get("yawn_gap_s", 0.5) if v is None else c.get("yawn_dip_s", 0.1)
            if (ts_ms - self._last_open) / 1000.0 > tol:
                self._finish(ts_ms)
        wms = c.get("yawn_window_s", 300) * 1000
        while self.yawns and self.yawns[0][0] < ts_ms - wms:
            self.yawns.popleft()

    def yawning(self, ts_ms: int) -> bool:
        """Sedang menguap (untuk overlay): run sudah >= yawn_min_s dan belum lewat yawn_max_s."""
        return self.in_run and self.c.get("yawn_min_s", 2.0) <= self.run_s(ts_ms) <= self.c.get("yawn_max_s", 10.0)


# ---------------------------------------------------------------- D-01 kantuk (A-02)
class D01Drowsy(Indicator):
    code = "D01"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.win = _TimeWindow(cfg.get("window_s", 30))
        self.closures: deque = deque()     # (t_end_ms, dur_s, t_start_ms) kedipan lambat
        self._closed_since: int | None = None
        self._last_closed = False
        self.yawn = _YawnTracker(cfg)
        self.score = 0.0                   # tingkat kantuk: >= 1 = ambang D-01

    def update(self, x):
        dt = self._dt(x.ts_ms)
        s = x.sig
        valid = s.face_present and s.eye_valid
        eye = s.eye_closed
        if self.c.get("closed_signal", "strict") == "strict" and s.eye_closed_strict is not None:
            eye = s.eye_closed_strict
        closed = valid and eye
        self.win.add(x.ts_ms, dt, (valid, closed))

        # deteksi kedipan lambat: durasi satu pejaman dalam [min, max]
        if closed and not self._last_closed:
            self._closed_since = x.ts_ms
        if not closed and self._last_closed and self._closed_since is not None:
            dur = (x.ts_ms - self._closed_since) / 1000.0
            lo, hi = self.c.get("slow_blink_s", [0.4, 1.4])
            if lo <= dur <= hi:
                self.closures.append((x.ts_ms, dur, self._closed_since))
            self._closed_since = None
        self._last_closed = closed
        wms = self.c.get("window_s", 30) * 1000
        while self.closures and self.closures[0][0] < x.ts_ms - wms:
            self.closures.popleft()

        span = self.win.span_s()
        valid_ratio = self.win.weighted_mean(lambda v: v[0])
        valid_time = sum(d for _, d, v in self.win.q if v[0])
        closed_time = sum(d for _, d, v in self.win.q if v[1])
        denom = valid_time
        if self.c.get("perclos_full_window", True):
            # penyebut minimal = jendela penuh x rasio valid minimum: jendela yang belum terisi (awal sesi)
            # tidak menggelembungkan PERCLOS -> satu pejaman 2 s tidak memicu D-01 sebelum D-02.
            denom = max(valid_time, self.c.get("window_s", 30) * self.c.get("min_valid_ratio", 0.7))
        perclos = closed_time / denom if denom > 0 else 0.0
        slow = len(self.closures)
        enough = span >= self.c.get("min_window_fill_s", 10)
        gates = enough and valid_ratio >= self.c.get("min_valid_ratio", 0.7)
        # jalur mata (PERCLOS / kedip lambat) -- TIDAK berubah oleh fitur menguap
        eye_active = bool(gates and (perclos > self.c.get("perclos_min", 0.20)
                                     or slow >= self.c.get("slow_blink_count", 4)))
        eye_score = max(perclos / max(1e-6, self.c.get("perclos_min", 0.2)),
                        slow / max(1, self.c.get("slow_blink_count", 4)))

        # jalur menguap: menambah skor kantuk; hanya aktif sebentar setelah menguap terakhir
        yawn_active = False
        n_yawn = 0
        bonus = 0.0
        if self.c.get("yawn_enabled", False):
            self.yawn.update(x.ts_ms, s)
            n_yawn = len(self.yawn.yawns)
            bonus = min(self.c.get("yawn_bonus_max", 0.4), self.c.get("yawn_weight", 0.2) * n_yawn)
            if n_yawn:
                recent = x.ts_ms - self.yawn.yawns[-1][0] <= self.c.get("yawn_active_hold_s", 5.0) * 1000
                alone = self.c.get("yawn_alone_count")
                yawn_active = recent and ((gates and eye_score + bonus >= 1.0)
                                          or (alone is not None and n_yawn >= alone))
        self.score = eye_score + bonus

        self.active = eye_active or yawn_active
        if self.active and self.condition_start_ms is None:
            if eye_active:
                first = self.closures[0][2] if self.closures else x.ts_ms
            else:
                first = self.yawn.yawns[-1][1]     # awal menguap yang memicu, bukan menit-menit lalu
            self.condition_start_ms = int(first)
        elif not self.active:
            self.condition_start_ms = None
        # aktif -> 1.0 (jalur mata aktif berarti skor >= 1 juga; jalur yawn_alone_count bisa < 1)
        self.confidence = 1.0 if self.active else min(1.0, self.score)
        self.info = {"perclos": round(perclos, 3), "slow_blinks": slow,
                     "valid_ratio": round(valid_ratio, 2), "span_s": round(span, 1)}
        if self.c.get("yawn_enabled", False):
            self.info.update({"yawns": n_yawn, "score": round(self.score, 2),
                              "yawning": self.yawn.yawning(x.ts_ms)})
        return self.active


# ---------------------------------------------------------------- D-04 kepala tertunduk / miring (A-03)
class D04HeadDrop(Indicator):
    code = "D04"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.nods: deque = deque()          # t_ms anggukan selesai
        self._down_since: int | None = None
        self.roll_run = 0.0
        self._roll_start: int | None = None
        self._first_nod_start: deque = deque()

    def update(self, x):
        dt = self._dt(x.ts_ms)
        s = x.sig
        c = self.c
        if s.face_present:
            if self._down_since is None and s.pitch < -c.get("nod_drop_deg", 15):
                self._down_since = x.ts_ms
            elif self._down_since is not None and s.pitch > -c.get("nod_return_deg", 8):
                dur = (x.ts_ms - self._down_since) / 1000.0
                if dur <= c.get("nod_max_s", 2.0):
                    self.nods.append(x.ts_ms)
                    self._first_nod_start.append(self._down_since)
                self._down_since = None
            if abs(s.roll) > c.get("roll_deg", 25):
                if self.roll_run == 0.0:
                    self._roll_start = x.ts_ms
                self.roll_run += dt
            else:
                self.roll_run = 0.0
                self._roll_start = None
        else:
            self.roll_run = 0.0
            self._roll_start = None
        wms = c.get("nod_window_s", 30) * 1000
        while self.nods and self.nods[0] < x.ts_ms - wms:
            self.nods.popleft()
            self._first_nod_start.popleft()
        nod_hit = len(self.nods) >= c.get("nod_count", 2)
        hold = c.get("nod_active_hold_s")
        if nod_hit and hold is not None and x.ts_ms - self.nods[-1] > hold * 1000:
            # opsi: kondisi anggukan dianggap selesai jika tidak ada anggukan baru selama `hold` detik
            nod_hit = False
        roll_hit = self.roll_run >= c.get("roll_s", 1.5)
        self.active = nod_hit or roll_hit
        if roll_hit:
            self.condition_start_ms = self._roll_start
        elif nod_hit:
            self.condition_start_ms = self._first_nod_start[0]
        else:
            self.condition_start_ms = None
        self.confidence = 1.0 if self.active else max(
            len(self.nods) / max(1, c.get("nod_count", 2)), self.roll_run / max(1e-6, c.get("roll_s", 1.5)))
        self.confidence = min(1.0, self.confidence)
        self.info = {"nods": len(self.nods), "roll_run_s": round(self.roll_run, 2)}
        return self.active


# ---------------------------------------------------------------- D-05 distraksi (A-02)
class D05Distraction(Indicator):
    code = "D05"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.win = _TimeWindow(cfg.get("window_s", 6))
        self.off_run = 0.0
        self._off_start: int | None = None
        self._pitch_only = False
        self._was_off = False
        self._last_off: tuple[bool, bool] | None = None   # status off-road terakhir saat wajah masih ada
        self._last_face_ms = 0
        self._now_ms = 0

    def off_road(self, s: FaceSignals, hyst_deg: float = 0.0) -> tuple[bool, bool]:
        """(off_road, karena_pitch_saja) untuk wajah yang terdeteksi."""
        by_yaw = abs(s.yaw) > self.c.get("yaw_deg", 30) - hyst_deg
        by_pitch = s.pitch < -(self.c.get("pitch_down_deg", 25) - hyst_deg)
        off = by_yaw or by_pitch
        if off and self.c.get("require_eyes_open", False) and s.eye_valid and s.eye_closed:
            off = False
        return off, (off and by_pitch and not by_yaw)

    def _classify(self, s: FaceSignals, blocked_frame: bool) -> tuple[bool, bool, bool]:
        """(off_mentah untuk akumulasi, off_dengan_hysteresis untuk run kontinu, karena_pitch_saja)"""
        if not s.face_present:
            # wajah hilang tepat setelah menoleh/menunduk jauh (mis. lihat pangkuan dalam-dalam):
            # status off-road terakhir dipertahankan sebentar. Tidak berlaku saat kamera terhalang (D-09).
            hold = self.c.get("face_lost_offroad_hold_s")
            if (hold is not None and not blocked_frame and self._last_off is not None
                    and self._now_ms - self._last_face_ms <= hold * 1000):
                return True, True, self._last_off[1]
            return False, False, False
        raw, raw_pitch_only = self.off_road(s)
        hyst, hyst_pitch_only = self.off_road(s, self.c.get("hysteresis_deg", 0.0) if self._was_off else 0.0)
        self._last_face_ms = self._now_ms
        self._last_off = (True, hyst_pitch_only) if hyst else None
        return raw, hyst, hyst_pitch_only

    def update(self, x):
        dt = self._dt(x.ts_ms)
        self._now_ms = x.ts_ms
        # akumulasi memakai ambang mentah (hysteresis menggeser rasio anggukan lambat -> D-05 palsu, risiko K4);
        # run kontinu memakai hysteresis supaya pitch yang berayun di sekitar ambang tidak memutus run.
        raw_off, off, pitch_only = self._classify(x.sig, x.blocked_frame)
        self._was_off = off
        self.win.add(x.ts_ms, dt, raw_off)
        if off:
            if self.off_run == 0.0:
                self._off_start = x.ts_ms
            self.off_run += dt
        else:
            self.off_run = 0.0
            self._off_start = None
        self._pitch_only = pitch_only
        need = self.c.get("continuous_s", 1.5)
        if pitch_only:
            need += self.c.get("pitch_only_extra_hold_s", 0.0)
        ratio = self.win.weighted_mean(lambda v: v)
        enough = self.win.span_s() >= self.c.get("min_window_fill_s", 3.0)
        cont_hit = self.off_run >= need
        acc_hit = enough and ratio >= self.c.get("ratio", 0.5)
        active = cont_hit or acc_hit
        if not x.moving:
            active = False
        if active and self.condition_start_ms is None:
            if cont_hit and self._off_start is not None:
                self.condition_start_ms = self._off_start
            else:
                first_off = next((t for t, _, v in self.win.q if v), x.ts_ms)
                self.condition_start_ms = int(first_off)
        elif not active:
            self.condition_start_ms = None
        self.active = active
        self.confidence = min(1.0, max(self.off_run / max(1e-6, need), ratio / max(1e-6, self.c.get("ratio", 0.5))))
        self.info = {"off_run_s": round(self.off_run, 2), "off_ratio": round(ratio, 2), "off": off}
        return self.active


# ---------------------------------------------------------------- D-06 ponsel (A-02)
class D06Phone(Indicator):
    code = "D06"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.win = _TimeWindow(cfg.get("window_s", 1.0))

    def update(self, x):
        dt = self._dt(x.ts_ms)
        if not x.yolo_valid:
            self.active = False
            self.info = {"yolo": "off"}
            return False
        hit = x.phone_conf >= self.c.get("conf", 0.35)
        self.win.add(x.ts_ms, dt, hit)
        ratio = self.win.weighted_mean(lambda v: v)
        enough = self.win.span_s() >= 0.5 * self.c.get("window_s", 1.0)
        active = enough and ratio >= self.c.get("ratio", 0.6)
        if not x.moving:
            active = False
        if active and self.condition_start_ms is None:
            self.condition_start_ms = int(next((t for t, _, v in self.win.q if v), x.ts_ms))
        elif not active:
            self.condition_start_ms = None
        self.active = active
        self.confidence = float(x.phone_conf)
        self.info = {"phone_ratio": round(ratio, 2), "phone_conf": round(x.phone_conf, 2)}
        return active


# ---------------------------------------------------------------- D-09 kamera terhalang (A-03)
class D09CameraBlock(Indicator):
    code = "D09"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.block_run = 0.0
        self._start: int | None = None

    def update(self, x):
        dt = self._dt(x.ts_ms)
        if x.blocked_frame and not x.sig.face_present:
            if self.block_run == 0.0:
                self._start = x.ts_ms
            self.block_run += dt
        else:
            self.block_run = 0.0
            self._start = None
        self.active = self.block_run >= self.c.get("hold_s", 1.0)
        self.condition_start_ms = self._start if self.active else None
        self.confidence = min(1.0, self.block_run / max(1e-6, self.c.get("hold_s", 1.0)))
        self.info = {"block_run_s": round(self.block_run, 2)}
        return self.active


# ---------------------------------------------------------------- D-10 APD (A-02)
class D10PPE(Indicator):
    code = "D10"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.win = _TimeWindow(cfg.get("window_s", 3.0))

    def update(self, x):
        dt = self._dt(x.ts_ms)
        if not x.yolo_valid:
            self.active = False
            self.info = {"yolo": "off"}
            return False
        self.win.add(x.ts_ms, dt, (x.no_helmet, x.no_vest))
        r_helmet = self.win.weighted_mean(lambda v: v[0])
        r_vest = self.win.weighted_mean(lambda v: v[1])
        enough = self.win.span_s() >= 0.8 * self.c.get("window_s", 3.0)
        thr = self.c.get("ratio", 0.7)
        active = enough and (r_helmet >= thr or r_vest >= thr)   # satu alert D-10 untuk helm dan/atau rompi
        if active and self.condition_start_ms is None:
            self.condition_start_ms = int(next((t for t, _, v in self.win.q if v[0] or v[1]), x.ts_ms))
        elif not active:
            self.condition_start_ms = None
        self.active = active
        self.confidence = max(r_helmet, r_vest)
        self.info = {"no_helmet_ratio": round(r_helmet, 2), "no_vest_ratio": round(r_vest, 2)}
        return active


def build_indicators(th: dict) -> dict[str, Indicator]:
    return {
        "D09": D09CameraBlock(th.get("D09_camera_block", {})),
        "D02": D02LongClosure(th.get("D02_long_closure", {})),
        "D04": D04HeadDrop(th.get("D04_head_drop", {})),
        "D06": D06Phone(th.get("D06_phone", {})),
        "D01": D01Drowsy(th.get("D01_drowsy", {})),
        "D05": D05Distraction(th.get("D05_distraction", {})),
        "D10": D10PPE(th.get("D10_ppe", {})),
    }
