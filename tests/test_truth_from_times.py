from pathlib import Path

from tools.truth_from_times import Segment, convert, to_epoch_ms

TZ = 7.0
DATE = "2026-11-24"


def _seg(name, start_hms, dur_s):
    first = to_epoch_ms(start_hms, DATE, TZ)
    return Segment(Path(name), first, first + dur_s * 1000)


def test_converts_wall_clock_to_video_seconds():
    segs = [_seg("cabin_a.avi", "15:30:00", 300), _seg("cabin_b.avi", "15:35:00", 300)]
    rows = [{"start": "15:31:05", "end": "15:32:05", "code": "D-01", "level": "A-02",
             "peraga": "P1", "cahaya": "terang", "scenario": "1"},
            {"start": "15:36:00", "end": "15:36:03", "code": "D02", "level": "A-03"}]
    out, problems = convert(rows, segs, DATE, TZ)
    assert problems == []
    assert out[0]["video"] == "cabin_a.avi" and out[0]["start_s"] == "65.00" and out[0]["end_s"] == "125.00"
    assert out[0]["expected_code"] == "D01"
    assert out[1]["video"] == "cabin_b.avi" and out[1]["start_s"] == "60.00"


def test_normal_driving_rows_and_missing_recording():
    segs = [_seg("cabin_a.avi", "15:30:00", 300)]
    rows = [{"start": "15:30:10", "end": "15:34:00", "code": "", "level": ""},
            {"start": "16:00:00", "end": "16:01:00", "code": "D05", "level": "A-02"}]
    out, problems = convert(rows, segs, DATE, TZ)
    assert len(out) == 1 and out[0]["expected_code"] == "NONE"
    assert any("tidak ada rekaman" in p for p in problems)


def test_scenario_past_segment_end_is_clipped():
    segs = [_seg("cabin_a.avi", "15:30:00", 60)]
    out, problems = convert([{"start": "15:30:50", "end": "15:31:20", "code": "D04", "level": "A-03"}],
                            segs, DATE, TZ)
    assert out[0]["end_s"] == "60.00" and any("dipotong" in p for p in problems)


def test_row_across_midnight():
    segs = [_seg("cabin_a.avi", "23:59:00", 300)]
    out, problems = convert([{"start": "23:59:30", "end": "00:00:30", "code": "D01", "level": "A-02"}],
                            segs, DATE, TZ)
    assert problems == [] and out[0]["start_s"] == "30.00" and out[0]["end_s"] == "90.00"


def test_clock_offset_shifts_rows():
    segs = [_seg("cabin_a.avi", "15:30:00", 300)]
    out, _ = convert([{"start": "15:31:00", "end": "15:31:03", "code": "D02", "level": "A-03"}],
                     segs, DATE, TZ, offset_s=3)         # jam Jetson 3 s lebih cepat
    assert out[0]["start_s"] == "63.00"


def test_relative_paths_under_out_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out" / "rec").mkdir(parents=True)
    first = to_epoch_ms("15:30:00", DATE, TZ)
    segs = [Segment(Path("out/rec/cabin_1.mp4"), first, first + 60_000)]
    out, _ = convert([{"start": "15:30:10", "end": "15:30:20", "code": "D02", "level": "A-03"}],
                     segs, DATE, TZ, base=(tmp_path / "out").resolve())
    assert out[0]["video"] == "rec/cabin_1.mp4"


def test_full_datetime_without_date_flag():
    assert to_epoch_ms("2026-11-24 15:30:00", None, TZ) == to_epoch_ms("15:30:00", DATE, TZ)
