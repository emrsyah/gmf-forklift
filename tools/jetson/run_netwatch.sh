#!/bin/bash
# Jalankan netwatch.py terus-menerus (restart jika berhenti). Dipasang lewat crontab user:
#   @reboot /home/jetson/gmf-forklift/tools/jetson/run_netwatch.sh
cd ~/gmf-forklift || exit 1
mkdir -p logs
while true; do
  python3 -u tools/jetson/netwatch.py --log-dir ~/gmf-forklift/logs >> logs/netwatch.log 2>&1
  sleep 5
done
