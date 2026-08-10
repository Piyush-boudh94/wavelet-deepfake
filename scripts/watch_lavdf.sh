#!/usr/bin/env bash
# Live LAV-DF preprocessing view. Run in a VS Code terminal:
#     ./scripts/watch_lavdf.sh
# Ctrl-C stops WATCHING only -- the job itself keeps running (it is nohup'd).
#
# The raw log's "2700/10000" is progress within the CURRENT class; "ok=" is the
# cumulative total across all six class/split folders. This reports the total.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

LOG=logs/preprocess_lavdf.log
TOTAL=30000            # train 20000 + val 4000 + test 6000

start_ok=""
start_t=$(date +%s)

while true; do
  line=$(grep -E '^  [0-9]+/[0-9]+  ok=' "$LOG" 2>/dev/null | tail -1)
  ok=$(sed -n 's/.*ok=\([0-9]*\).*/\1/p' <<<"$line")
  fail=$(sed -n 's/.*fail=\([0-9]*\).*/\1/p' <<<"$line")
  ok=${ok:-0}; fail=${fail:-0}
  [ -z "$start_ok" ] && start_ok=$ok

  now=$(date +%s); el=$((now - start_t))
  rate=0; eta="--"
  if [ "$el" -gt 30 ] && [ "$ok" -gt "$start_ok" ]; then
    rate=$(( (ok - start_ok) * 60 / el ))            # videos/min this session
    [ "$rate" -gt 0 ] && eta=$(( (TOTAL - ok) / rate ))
  fi

  pct=$(( ok * 100 / TOTAL ))
  filled=$(( pct * 40 / 100 ))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((40 - filled)) '')

  frames=$(find data/processed/lavdf -name '*.png' 2>/dev/null | wc -l)
  alive=$(pgrep -f preprocess.py >/dev/null && echo RUNNING || echo "STOPPED !!")

  clear
  echo "LAV-DF preprocessing          $(date '+%H:%M:%S')"
  echo "-------------------------------------------------------"
  echo "  state    : $alive"
  echo "  videos   : ${ok} / ${TOTAL}   (failed: ${fail})"
  echo "  [${bar}] ${pct}%"
  echo "  frames   : ${frames}"
  echo "  rate     : ${rate} videos/min"
  echo "  ETA      : ${eta} min"
  echo "  now on   : ${line:-starting...}"
  echo "-------------------------------------------------------"
  echo "  Ctrl-C stops this view, NOT the job."
  sleep 10
done
