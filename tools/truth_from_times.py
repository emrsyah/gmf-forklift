"""Catatan jam skenario (sesi uji di forklift) -> truth.csv untuk replay harness.

Di forklift tidak ada tombol penanda seperti tools/recorder.py. Rekaman kabin mentah
(`continuous_recording`, segmen + <video>.ts.csv) jalan terus, dan pencatat menulis jam mulai/selesai
setiap peragaan. Skrip ini mencari segmen yang memuat jam itu dan menghitung detik video-nya
(sama dengan VideoFileSource: ts frame - ts frame pertama).

times.csv (jam lokal, zona `timezone_offset_h` di runtime.yaml; tanggal boleh di kolom atau --date):
  start,end,code,level,peraga,cahaya,scenario
  15:31:05,15:32:05,D01,A-02,P1,terang,1
  15:40:00,15:50:00,NONE,,P1,terang,normal      <- mengemudi normal (alarm palsu per jam)

  python -m tools.truth_from_times --times sesi.csv --recordings recordings/cabin --date 2026-11-24 \
      --out eval/scenarios/truth_forklift.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.capture import load_sidecar_ts
from src.config import Config

VIDEO_EXT = (".mp4", ".avi", ".mkv")
FIELDS = ["video", "start_s", "end_s", "expected_code", "expected_level", "peraga", "cahaya", "scenario"]


@dataclass
class Segment:
    path: Path
    first_ms: int
    last_ms: int


def load_segments(rec_dir: Path) -> list[Segment]:
    segs = []
    for p in sorted(rec_dir.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        first, rel = load_sidecar_ts(p)
        if first is None:
            continue
        segs.append(Segment(p, first, first + int(round(rel[-1] * 1000))))
    return segs


def to_epoch_ms(text: str, date: str | None, tz_h: float) -> int:
    text = text.strip()
    if len(text) <= 8:                       # HH:MM[:SS]
        if not date:
            raise ValueError(f"jam '{text}' tanpa tanggal: isi --date atau tulis 'YYYY-MM-DD HH:MM:SS'")
        text = f"{date} {text}"
    fmt = "%Y-%m-%d %H:%M:%S" if text.count(":") == 2 else "%Y-%m-%d %H:%M"
    dt = datetime.strptime(text, fmt).replace(tzinfo=timezone(timedelta(hours=tz_h)))
    return int(dt.timestamp() * 1000)


def convert(rows: list[dict], segs: list[Segment], date: str | None, tz_h: float,
            base: Path | None = None, offset_s: float = 0.0) -> tuple[list[dict], list[str]]:
    """offset_s: jam Jetson - jam pencatat (detik), mis. +3 jika jam Jetson 3 s lebih cepat."""
    out, problems = [], []
    off_ms = int(round(offset_s * 1000))
    for i, r in enumerate(rows, 1):
        s_ms = to_epoch_ms(r["start"], date, tz_h) + off_ms
        e_ms = to_epoch_ms(r["end"], date, tz_h) + off_ms
        if e_ms <= s_ms and len(r["end"].strip()) <= 8:   # jam saja, melewati tengah malam
            e_ms += 24 * 3600 * 1000
        if e_ms <= s_ms:
            problems.append(f"baris {i}: end <= start")
            continue
        seg = next((g for g in segs if g.first_ms <= s_ms <= g.last_ms), None)
        if seg is None:
            problems.append(f"baris {i}: tidak ada rekaman yang memuat {r['start']}")
            continue
        if e_ms > seg.last_ms:   # skenario melewati batas segmen: potong (sisanya di segmen berikut tidak dinilai)
            problems.append(f"baris {i}: melewati akhir {seg.path.name}, dipotong")
            e_ms = seg.last_ms
        video = seg.path.resolve().relative_to(base) if base else seg.path
        out.append({
            "video": video.as_posix(),
            "start_s": f"{(s_ms - seg.first_ms) / 1000:.2f}",
            "end_s": f"{(e_ms - seg.first_ms) / 1000:.2f}",
            "expected_code": (r.get("code") or "NONE").strip().upper().replace("D-", "D"),
            "expected_level": (r.get("level") or "").strip(),
            "peraga": (r.get("peraga") or "").strip(),
            "cahaya": (r.get("cahaya") or "").strip(),
            "scenario": (r.get("scenario") or "").strip(),
        })
    return out, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--times", required=True)
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD bila kolom jam hanya HH:MM:SS")
    ap.add_argument("--offset-s", type=float, default=0.0,
                    help="jam Jetson - jam pencatat (detik); cek dengan tanda sinkron jam di awal sesi")
    a = ap.parse_args()
    tz_h = float(Config.load().runtime.get("timezone_offset_h", 7.0))
    with open(a.times, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out_path = Path(a.out)
    segs = load_segments(Path(a.recordings))
    out, problems = convert(rows, segs, a.date, tz_h, base=out_path.parent.resolve()
                            if all(s.path.resolve().is_relative_to(out_path.parent.resolve()) for s in segs) else None,
                            offset_s=a.offset_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)}/{len(rows)} baris -> {out_path} ({len(segs)} segmen rekaman)")
    for p in problems:
        print("  !", p)


if __name__ == "__main__":
    main()
