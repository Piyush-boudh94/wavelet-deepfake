#!/usr/bin/env bash
# Patiently fetch the remaining DFDC videos through Kaggle's rate limiter.
#
# History that shaped this: 6 parallel workers earned a blanket account-wide
# HTTP 429 (2,592 consecutive failures). A later retry at 2 workers ALSO failed
# every file -- the ban outlives a single 10-minute wait. So this loop:
#
#   1. probes ONCE, cheaply, before attempting anything
#   2. runs the downloader, which trips its own circuit breaker (exit 75)
#      after 15 consecutive 429s instead of grinding thousands of requests
#   3. backs off for progressively longer between attempts
#
# The downloader is resumable and skips whatever is already on disk, so being
# interrupted at any point costs nothing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"

PROBE_FILE="dfdc_train_part_03/dfdc_train_part_3/iumybqkbzt.mp4"
WAITS=(900 1800 3600 3600 7200 7200)   # 15m, 30m, 1h, 1h, 2h, 2h...
round=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

while true; do
  target=$(ls data/raw/dfdc/_incoming/*.mp4 2>/dev/null | wc -l)
  log "attempt $((round + 1)): ${target} files staged so far"

  probe_dir=$(mktemp -d)
  out=$(kaggle datasets download pranay22077/dfdc-10 -f "$PROBE_FILE" \
          -p "$probe_dir" --force 2>&1)
  rm -rf "$probe_dir"

  if grep -q "429\|Too Many Requests" <<<"$out"; then
    log "still rate-limited (probe)"
  else
    log "probe clear -- starting downloader (1 worker, 2s spacing)"
    MIN_INTERVAL_OVERRIDE=2.0 python3 -u scripts/download_dfdc.py --workers 1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "DOWNLOAD COMPLETE"
      exit 0
    fi
    log "downloader exited $rc (rate-limited again)"
  fi

  idx=$(( round < ${#WAITS[@]} ? round : ${#WAITS[@]} - 1 ))
  wait_s=${WAITS[$idx]}
  log "backing off ${wait_s}s"
  sleep "$wait_s"
  round=$((round + 1))
done
