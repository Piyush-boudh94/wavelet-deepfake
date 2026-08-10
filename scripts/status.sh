#!/usr/bin/env bash
# One-glance status of every background job. Run bare, or live:
#     watch -n 10 ./scripts/status.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

NS=dgx-s-bmu-cse-240577-restricted
bar() { printf '%*s\n' 64 '' | tr ' ' '-'; }

pct() {  # pct <done> <total>
  [ "${2:-0}" -gt 0 ] || { echo "0"; return; }
  echo $(( $1 * 100 / $2 ))
}

bar
echo "WMamba status  --  $(date '+%Y-%m-%d %H:%M:%S')"
bar

if pgrep -f supervisor.sh >/dev/null; then
  echo "SUPERVISOR : ACTIVE (self-healing on; $(grep -c RESTART logs/supervisor.log 2>/dev/null | head -1) restarts so far)"
else
  echo "SUPERVISOR : OFF  -- jobs will NOT restart after a pod wipe"
  echo "             start: nohup ./scripts/supervisor.sh > logs/supervisor.log 2>&1 &"
fi
echo

# ---------------------------------------------------------------- DFDC download
echo "DFDC DOWNLOAD"
if pgrep -f download_dfdc.py >/dev/null; then
  echo "  state   : DOWNLOADING (pid $(pgrep -f download_dfdc.py | head -1))"
elif pgrep -f download_dfdc_when_clear >/dev/null; then
  echo "  state   : WAITING OUT KAGGLE RATE LIMIT (auto-resumes when clear)"
  grep -E '^\[[0-9:]+\] probe' logs/download_dfdc.log 2>/dev/null | tail -1 | sed 's/^/  last probe:/'
else
  echo "  state   : NOT RUNNING"
fi
staged=$(ls data/raw/dfdc/_incoming/*.mp4 2>/dev/null | wc -l)
echo "  staged  : ${staged} files, $(du -sh data/raw/dfdc/_incoming 2>/dev/null | cut -f1 || echo 0)"
grep -E '^  [0-9]+/[0-9]+' logs/download_dfdc.log 2>/dev/null | tail -1 | sed 's/^/  progress:/'

# ------------------------------------------------------------ LAV-DF preprocess
echo
echo "LAV-DF PREPROCESSING"
if pgrep -f "preprocess.py" >/dev/null; then
  echo "  state   : RUNNING"
else
  echo "  state   : not running"
fi
frames=$(find data/processed/lavdf -name '*.png' 2>/dev/null | wc -l)
nshards=$(pgrep -fc 'shard [0-9]+/[0-9]+' 2>/dev/null | head -1)
echo "  shards  : ${nshards:-0} running"
echo "  frames  : ${frames}"
# total videos = union across every shard manifest (they never overlap)
done_v=$(python3 - <<'EOF' 2>/dev/null
import json,glob
ids=set()
for f in glob.glob('data/processed/lavdf/manifest*.json'):
    try:
        ids.update(e['video_id'] for e in json.load(open(f))['entries'])
    except Exception: pass
print(len(ids))
EOF
)
echo "  videos  : ${done_v:-?} / 30000"

# ------------------------------------------------------------------- training
echo
echo "LAV-DF TRAINING"
if pgrep -f "src.training.train" >/dev/null; then
  echo "  state   : TRAINING"
  grep -E 'epoch [0-9]+ (done|val_auc)|EARLY STOP|SMOKE' logs/train_lavdf.log 2>/dev/null | tail -2 | sed 's/^/  /'
elif pgrep -f train_when_ready >/dev/null; then
  echo "  state   : ARMED -- auto-starts when preprocessing finishes"
  tail -1 logs/train_lavdf.log 2>/dev/null | sed 's/^/  last    : /'
else
  echo "  state   : not armed"
fi

# ------------------------------------------------------------------ dataset now
echo
echo "DATASETS ON DISK"
for ds in dfdc lavdf; do
  n=$(find "data/raw/$ds" -name '*.mp4' 2>/dev/null | wc -l)
  echo "  raw/$ds : ${n} videos, $(du -sh "data/raw/$ds" 2>/dev/null | cut -f1)"
done

# -------------------------------------------------------------------- cluster
echo
echo "CLUSTER"
pod=$(kubectl get pods -n "$NS" --no-headers 2>/dev/null | head -1)
echo "  pod     : ${pod:-NONE (run: kubectl apply -f scripts/k8s/dev-pod.yaml)}"
echo "  disk    : $(df -h /home/dgx-s-bmu-cse-240577 | tail -1 | awk '{print $4" free of "$2}')"
bar
