#!/bin/bash
# Uji kecepatan jalur 5G (lewat interface modem saja), satu baris JSON ke stdout dan ke
# ~/gmf-forklift/logs/netspeed-YYYYMMDD.jsonl.
#   upload/download : Mbps ke speed.cloudflare.com (data acak, tanpa isi apa pun dari perangkat)
#   latency         : TCP connect + TLS ke server MINERVA (yang dipakai agen untuk alarm, frame, klip)
# Ukuran kecil supaya tidak mengganggu alarm lama-lama:  netspeed.sh [UP_MB] [DOWN_MB]
UP_MB="${1:-4}"
DOWN_MB="${2:-10}"
SERVER="${NETSPEED_SERVER:-https://minerva.43-134-5-10.sslip.io}"
IFACE="$(ip -br link | awk '/^enx/{print $1; exit}')"
LOG_DIR="$HOME/gmf-forklift/logs"
mkdir -p "$LOG_DIR"
TMP="$(mktemp)"
head -c "$((UP_MB * 1000000))" /dev/urandom > "$TMP"

c=(curl -s -o /dev/null --interface "$IFACE" --max-time 25)
up=$("${c[@]}" -w '%{speed_upload} %{size_upload} %{http_code}' --data-binary @"$TMP" \
      -H 'Content-Type: application/octet-stream' https://speed.cloudflare.com/__up)
down=$("${c[@]}" -w '%{speed_download} %{size_download} %{http_code}' \
      "https://speed.cloudflare.com/__down?bytes=$((DOWN_MB * 1000000))")
lat=$("${c[@]}" -w '%{time_connect} %{time_appconnect} %{http_code}' "$SERVER/")
rm -f "$TMP"

python3 - "$IFACE" "$up" "$down" "$lat" "$LOG_DIR" <<'EOF'
import json, sys, time
iface, up, down, lat, log_dir = sys.argv[1:6]
u, d, l = up.split(), down.split(), lat.split()
mbps = lambda bps: round(float(bps) * 8 / 1e6, 2)
row = {
    "ts_ms": int(time.time() * 1000), "iface": iface,
    "up_mbps": mbps(u[0]) if len(u) == 3 and u[2] == "200" else None, "up_bytes": int(float(u[1])) if len(u) == 3 else 0,
    "down_mbps": mbps(d[0]) if len(d) == 3 and d[2] == "200" else None, "down_bytes": int(float(d[1])) if len(d) == 3 else 0,
    "server_tcp_ms": round(float(l[0]) * 1000, 1) if len(l) == 3 and float(l[0]) > 0 else None,
    "server_tls_ms": round(float(l[1]) * 1000, 1) if len(l) == 3 and float(l[1]) > 0 else None,
}
line = json.dumps(row, separators=(",", ":"))
print(line)
with open(time.strftime(f"{log_dir}/netspeed-%Y%m%d.jsonl"), "a") as f:
    f.write(line + "\n")
EOF
