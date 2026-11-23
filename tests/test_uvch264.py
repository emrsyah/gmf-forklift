"""Parser H.264-di-MJPG dan pencarian extension unit (tanpa kamera; jalan di Windows juga)."""
import struct
import uuid

from src.capture.uvch264 import (CFG_SIZE, UVCX_GUID, extract_h264, find_xu_unit, is_idr, nal_types,
                                 nal_units, pack_config, unpack_config)

SPS = b"\x00\x00\x00\x01\x67\x42\xc0\x1f\xaa"
PPS = b"\x00\x00\x00\x01\x68\xce\x3c\x80"
IDR = b"\x00\x00\x01\x65" + b"\x88" * 50
P = b"\x00\x00\x00\x01\x41" + b"\x9a" * 30


def _seg(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def _app4_header(w=1280, h=720) -> bytes:
    # bcdVersion, wHeaderLen=22, fourcc, wWidth, wHeight, dwFrameInterval, wDelay, dwPTS
    return struct.pack("<HH4sHHIHI", 0x0100, 22, b"H264", w, h, 333333, 0, 12345)


def _jpeg(app4_parts: list[bytes]) -> bytes:
    out = b"\xff\xd8" + _seg(0xE0, b"JFIF\x00\x01\x01" + b"\x00" * 7)
    for part in app4_parts:
        out += _seg(0xE4, part)
    out += _seg(0xDB, b"\x00" * 65)                    # DQT
    out += _seg(0xDA, b"\x01\x01\x00\x00\x3f\x00")     # SOS
    out += b"\x12\x34\xff\x00\x56" + b"\xff\xd9"       # data entropi (dengan byte stuffing) + EOI
    return out


def test_extract_single_app4():
    au = SPS + PPS + IDR
    jpg = _jpeg([_app4_header() + struct.pack("<I", len(au)) + au])
    assert extract_h264(jpg) == au


def test_extract_payload_split_over_segments():
    au = SPS + PPS + IDR + P * 3
    blob = _app4_header() + struct.pack("<I", len(au)) + au
    jpg = _jpeg([blob[:40], blob[40:90], blob[90:]])
    assert extract_h264(jpg) == au


def test_extract_none_without_app4_or_not_jpeg():
    assert extract_h264(_jpeg([])) is None
    assert extract_h264(b"not a jpeg") is None
    assert extract_h264(_jpeg([b"XXXXjunk"])) is None


def test_nal_parsing_3_and_4_byte_start_codes():
    au = SPS + PPS + IDR
    assert [t for t, _, _ in nal_units(au)] == [7, 8, 5]
    s, e = nal_units(au)[0][1:]
    assert au[s:e] == SPS
    assert nal_types(au) == {5, 7, 8}
    assert is_idr(au) and not is_idr(P)


def _iface(cls, sub) -> bytes:
    return bytes([9, 0x04, 0, 0, 0, cls, sub, 0, 0])


def _xu(unit: int, guid: uuid.UUID) -> bytes:
    body = bytes([0x24, 0x06, unit]) + guid.bytes_le + bytes([11, 1, 3, 2, 0xFF, 0x07, 0])
    return bytes([len(body) + 1]) + body


def test_find_xu_unit_from_descriptors():
    other = uuid.UUID("23e49ed0-1178-4f31-ae52-d2fb8a8d3b48")
    desc = (bytes([18, 0x01]) + b"\x00" * 16          # device
            + bytes([9, 0x02]) + b"\x00" * 7          # configuration
            + _iface(0x0E, 0x01) + _xu(6, other) + _xu(12, UVCX_GUID)
            + _iface(0x0E, 0x02)
            # VS_FORMAT_MJPEG juga subtype 0x06 -> harus diabaikan (bukan interface video control)
            + bytes([11, 0x24, 0x06, 1, 1, 1, 0, 0, 0, 0, 0]))
    assert find_xu_unit(desc) == 12
    assert find_xu_unit(desc, uuid.uuid4()) is None


def test_config_roundtrip():
    raw = bytes(range(CFG_SIZE))
    assert pack_config(unpack_config(raw)) == raw
