#!/usr/bin/env bash
# Run LAV-DF preprocessing as N parallel shards, to ACTUAL completion.
#
# Two failures this script now defends against, both of which cost real GPU time:
#
# 1. COLD-CACHE RACE. torch.hub caches RetinaFace as a .zip and extracts it to a
#    .pth on first load. Six shards starting together all performed that
#    extraction at once; some read a half-written .pth and died with
#    "EOFError: Ran out of input". Five of six shards were gone within seconds.
#    -> the model is now loaded ONCE, alone, before any shard starts.
#
# 2. "MERGED" MISTAKEN FOR "FINISHED". The launcher merged whenever the shards
#    exited -- for any reason, including crashing -- and the supervisor treats a
#    merge line as proof preprocessing is done. Training then started on 1,666
#    of 17,387 fake videos (8.7% fake instead of ~50%).
#    -> completion is now verified against selection.json, and shards are
#       relaunched until the counts actually match.
#
#   ./scripts/run_preprocess_shards.sh [N] [MAX_PASSES]
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

N=${1:-6}
MAX_PASSES=${2:-40}
ARGS='--dataset lavdf --subsample "train=20000,val=4000,test=6000" --train-frames 16 --eval-frames 32'

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if pgrep -f "preprocess.py --dataset lavdf" >/dev/null; then
  log "preprocessing already running -- not starting more"
  exit 0
fi

# ---- 1. give every shard its OWN weights copy ------------------------------
# ~/.cache is head-node XFS re-exported to the pod over NFS. Six shards reading
# the same 100 MB checkpoint through it failed with "PytorchStreamReader failed
# reading file ... file read failed" and, before that, a truncated-extraction
# "EOFError: Ran out of input". Pre-warming a shared cache was not enough --
# concurrent NFS reads of one big file are the problem. 200 MB per shard buys
# complete independence.
log "ensuring per-shard RetinaFace caches"
for i in $(seq 0 $((N - 1))); do
  d="$HOME/.cache/torch_shard${i}/hub/checkpoints"
  mkdir -p "$d"
  cp -n "$HOME/.cache/torch/hub/checkpoints/"* "$d/" 2>/dev/null || true
done

# ---- 2. shard passes until the selection is genuinely satisfied -------------
NS=dgx-s-bmu-cse-240577-restricted

# Without this the retry loop is useless when the namespace is wiped: every
# shard fails instantly with "No running 'wmamba-dev' pod", so all 10 passes
# burned in 12 SECONDS and the launcher gave up while the real fix was simply
# to wait for the pod. Passes must be gated on the pod, not on a counter.
wait_for_pod() {
  kubectl get pods -n "$NS" --no-headers 2>/dev/null | grep -q Running && return 0
  log "pod missing -- recreating and waiting"
  kubectl apply -f scripts/k8s/dev-pod.yaml >/dev/null 2>&1
  for _ in $(seq 1 120); do
    sleep 5
    kubectl get pods -n "$NS" --no-headers 2>/dev/null | grep -q Running && {
      log "pod is Running again"; return 0; }
  done
  log "WARN: pod did not return within 10 min"
  return 1
}

for pass_no in $(seq 1 "$MAX_PASSES"); do
  wait_for_pod || { sleep 60; continue; }
  pass_start=$(date +%s)
  log "pass ${pass_no}/${MAX_PASSES}: launching $N shards"
  pids=()
  for i in $(seq 0 $((N - 1))); do
    nohup bash -c "./scripts/pod.sh exec 'TORCH_HOME=\$HOME/.cache/torch_shard${i} .venv/bin/python -u scripts/preprocess.py ${ARGS} --shard ${i}/${N}'" \
      >> "logs/preprocess_lavdf_shard${i}.log" 2>&1 &
    pids+=($!)
    sleep 2
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "pass ${pass_no}: all shards exited"

  # merge first, so the completeness check sees every shard's work
  ./scripts/pod.sh exec ".venv/bin/python scripts/preprocess.py --dataset lavdf --merge" \
     2>&1 | tail -5 | tee -a logs/preprocess_lavdf.log

  if ./scripts/pod.sh exec ".venv/bin/python scripts/preprocess.py --dataset lavdf --verify" 2>&1 | tee -a logs/preprocess_lavdf.log | grep -q "PREPROCESSING COMPLETE"; then
    log "PREPROCESSING VERIFIED COMPLETE after ${pass_no} pass(es)"
    exit 0
  fi
  # A pass that ends almost immediately did no work (pod gone, weights
  # unreadable). Pause so the retry budget is not spent in seconds.
  elapsed=$(( $(date +%s) - pass_start ))
  if [ "$elapsed" -lt 60 ]; then
    log "pass ${pass_no} lasted ${elapsed}s -- infrastructure problem, backing off 120s"
    sleep 120
  fi
  log "pass ${pass_no}: still incomplete -- relaunching (shards resume, no work redone)"
done

log "ERROR: still incomplete after ${MAX_PASSES} passes -- see logs/preprocess_lavdf_shard*.log"
exit 1
