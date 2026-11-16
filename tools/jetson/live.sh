#!/usr/bin/env bash
# Kendali runtime kabin di Jetson tanpa keyboard/mouse (lewat SSH).
#   tools/jetson/live.sh start [argumen src.main tambahan...]   # tampil di monitor Jetson (DISPLAY :0)
#   tools/jetson/live.sh key c        # kirim hotkey: c = kalibrasi ulang, d = debug, p = populasi, q = keluar
#   tools/jetson/live.sh stop
#   tools/jetson/live.sh log          # status kalibrasi, event, FPS terakhir
set -u
cd "$(dirname "$0")/../.."
KEYS=/tmp/gmf_keys.txt
LOG=/tmp/gmf_live.log
PIDF=/tmp/gmf_live.pid
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

running() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "${1:-}" in
  start)
    shift
    if running; then echo "sudah jalan (pid $(cat "$PIDF"))"; exit 0; fi
    : > "$KEYS"
    # shellcheck disable=SC1091
    . .venv/bin/activate
    setsid bash -c "tail -n0 -f '$KEYS' | exec python -u -m src.main --overlay configs/jetson.yaml --stdin-keys --log-fps 5 $*" \
      > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    echo "mulai (pid $(cat "$PIDF")), log: $LOG"
    ;;
  key)
    echo "${2:?hotkey, mis. c}" >> "$KEYS"
    ;;
  stop)
    if running; then
      echo q >> "$KEYS"; sleep 3
      running && kill -- -"$(cat "$PIDF")" 2>/dev/null
    fi
    rm -f "$PIDF"; echo "berhenti"
    ;;
  log)
    grep -E "kalibrasi|event|\[fps\]|\[wajah\]|\[kontrol\]|Error|rror" "$LOG" | tail -n "${2:-15}"
    ;;
  *)
    echo "pakai: $0 start|key <c|d|p|q>|stop|log [n]"; exit 1
    ;;
esac
