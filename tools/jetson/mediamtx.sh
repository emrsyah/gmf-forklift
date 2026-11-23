#!/usr/bin/env bash
# MediaMTX (relay RTSP/WebRTC kamera depan) di Jetson, tanpa systemd.
#   tools/jetson/mediamtx.sh start | stop | status | log
# Binary dicari di $MEDIAMTX_BIN, lalu ~/mediamtx/mediamtx, lalu PATH. Binary tidak di-commit;
# unduh sendiri rilis linux_arm64 dari https://github.com/bluenviron/mediamtx/releases.
# Setelah jalan: set front_preview.stream.enabled: true, buka http://<jetson>:8889/front.
set -u
cd "$(dirname "$0")/../.."
CONF="$(pwd)/tools/jetson/mediamtx.yml"
LOG=/tmp/gmf_mediamtx.log
PIDF=/tmp/gmf_mediamtx.pid

find_bin() {
  for b in "${MEDIAMTX_BIN:-}" "$HOME/mediamtx/mediamtx" "$(command -v mediamtx 2>/dev/null)"; do
    [ -n "$b" ] && [ -x "$b" ] && { echo "$b"; return 0; }
  done
  return 1
}
running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "${1:-}" in
  start)
    if running; then echo "sudah jalan (pid $(cat "$PIDF"))"; exit 0; fi
    BIN="$(find_bin)" || { echo "binary mediamtx tidak ditemukan (set MEDIAMTX_BIN)"; exit 1; }
    # nice: relay kalah prioritas dari deteksi (K5)
    setsid nice -n 5 "$BIN" "$CONF" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    sleep 1
    if running; then echo "mulai (pid $(cat "$PIDF")), log: $LOG"; else echo "gagal:"; tail -n 20 "$LOG"; exit 1; fi
    ;;
  stop)
    running && kill "$(cat "$PIDF")" 2>/dev/null
    rm -f "$PIDF"; echo "berhenti"
    ;;
  status)
    if running; then echo "jalan (pid $(cat "$PIDF"))"; else echo "mati"; fi
    ;;
  log)
    tail -n "${2:-30}" "$LOG"
    ;;
  *)
    echo "pakai: $0 start|stop|status|log [n]"; exit 1
    ;;
esac
