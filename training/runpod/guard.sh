#!/usr/bin/env bash
# Penjaga biaya independen (jalan di tmux terpisah dari training):
#   - jika <OUT>/STAGE berisi DONE -> tunggu GRACE_MIN lalu stop pod,
#   - batas keras: stop pod setelah DEADLINE_H jam apa pun yang terjadi.
# Stop (bukan terminate) -> volume /workspace (hasil + tarball) tetap ada untuk diambil nanti.
# Kredensial = kunci pod sendiri dari /proc/1/environ (disuntik RunPod), bukan kunci pengguna.
#   tmux new -d -s guard 'OUT=/workspace/out_cabin_v2 DEADLINE_H=5 GRACE_MIN=10 bash training/runpod/guard.sh'
set -u
OUT=${OUT:-/workspace/out_cabin_v2}
DEADLINE_H=${DEADLINE_H:-5}
GRACE_MIN=${GRACE_MIN:-10}
while IFS= read -r -d '' kv; do
  case "$kv" in RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "${kv?}";; esac
done < /proc/1/environ
stop_pod() {
  echo "[guard] $(date -u +%H:%M:%S) $1 -> podStop"
  curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}"
  echo
}
end=$(( $(date +%s) + DEADLINE_H * 3600 ))
while true; do
  if [ "$(cat "$OUT/STAGE" 2>/dev/null)" = "DONE" ]; then
    sleep $((GRACE_MIN * 60)); stop_pod "DONE + ${GRACE_MIN} min"; sleep 600
  fi
  if [ "$(date +%s)" -ge "$end" ]; then stop_pod "deadline ${DEADLINE_H} h"; sleep 600; fi
  sleep 60
done
