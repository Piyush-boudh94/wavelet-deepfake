#!/usr/bin/env bash
# Self-healing supervisor for the whole WMamba pipeline.
#
# WHY: this namespace gets deleted externally ~2x/day (pod, Deployment and all).
# Every long job here is individually resumable, but something has to notice the
# pod died and restart things. That was a human. Now it is this.
#
# Every CHECK_EVERY seconds it reconciles reality against what should be running:
#   1. pod missing            -> kubectl apply, wait for Running
#   2. preprocessing stalled  -> relaunch (skips completed videos)
#   3. training stalled       -> relaunch (auto-resumes from last checkpoint)
#   4. DFDC download stalled  -> relaunch (skips files already on disk)
#
# Every action is idempotent and every underlying script resumes, so a spurious
# restart costs seconds, never data. Safe to run forever.
#
#   start : nohup ./scripts/supervisor.sh > logs/supervisor.log 2>&1 &
#   stop  : touch STOP_SUPERVISOR      (or pkill -f supervisor.sh)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

NS=dgx-s-bmu-cse-240577-restricted
CHECK_EVERY=60
STOP_FILE=STOP_SUPERVISOR
# 6 shards measured at 108 videos/min vs 50 single-process (2.2x). Do NOT raise:
# the pod's 16-CPU quota is already saturated (load avg 19.5), so more shards
# thrash rather than help. The GPU is NOT the limit here (31% VRAM at 6 shards).
SHARDS=6
LOGT=logs/train_lavdf.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

running() { pgrep -f "$1" >/dev/null 2>&1; }

pod_ready() {
  kubectl get pods -n "$NS" --no-headers 2>/dev/null | grep -q "Running"
}

ensure_pod() {
  pod_ready && return 0
  log "POD MISSING -- recreating from scripts/k8s/dev-pod.yaml"
  kubectl apply -f scripts/k8s/dev-pod.yaml >/dev/null 2>&1
  for _ in $(seq 1 60); do
    sleep 5
    if pod_ready; then
      log "pod is Running again"
      return 0
    fi
  done
  log "WARN: pod did not become Running within 5 min; will retry next cycle"
  return 1
}

preprocessing_done() {
  # Completion is a property of the DATA, not of a log line. Grepping for
  # "merged" was wrong: the launcher merges whenever shards exit, crashes
  # included, so a crashed run looked finished and training began on 1,666 of
  # 17,387 fake videos. --verify compares the manifest against selection.json.
  ./scripts/pod.sh exec \
    ".venv/bin/python scripts/preprocess.py --dataset lavdf --verify" \
    2>/dev/null | grep -q "PREPROCESSING COMPLETE"
}

training_done() {
  grep -qE "training complete|EARLY STOP" "$LOGT" 2>/dev/null
}

# Has ANY epoch beyond the first ever finished? Distinguishes "crashing in the
# same place forever" from "progressing but interrupted by pod wipes".
epoch_completed() {
  grep -qE "epoch [1-9][0-9]* done" "$LOGT" 2>/dev/null
}

download_done() {
  grep -q "DOWNLOAD COMPLETE" logs/download_dfdc.log 2>/dev/null
}

log "supervisor up (pid $$). stop with: touch $STOP_FILE"
restarts=0
train_restarts=0
loop_warned=0

while [ ! -f "$STOP_FILE" ]; do

  # ---------------------------------------------------------------- 1. the pod
  # Everything GPU-side runs through it, so heal this first and skip the rest of
  # the cycle if it is not back yet.
  if ! pod_ready; then
    ensure_pod || { sleep "$CHECK_EVERY"; continue; }
  fi

  # ------------------------------------------------------ 2. LAV-DF preprocessing
  if ! preprocessing_done; then
    if ! running "preprocess.py --dataset lavdf"; then
      restarts=$((restarts + 1))
      log "RESTART #${restarts}: LAV-DF preprocessing, ${SHARDS} shards (resumes)"
      nohup ./scripts/run_preprocess_shards.sh "$SHARDS" \
        >> logs/preprocess_lavdf.log 2>&1 &
      sleep 30
    fi

  # --------------------------------------------------------- 3. LAV-DF training
  # Only once preprocessing has genuinely finished. train_when_ready.sh does the
  # first launch with preflight; the supervisor covers relaunches after a wipe,
  # where the checkpoint makes preflight redundant.
  elif ! training_done; then
    if ! running "src.training.train" && ! running "train_when_ready"; then

      # LOOP BREAKER. On 2026-08-04 this supervisor restarted training 55 times
      # over 6 h while it crashed at the end of every epoch 0 (val split had a
      # single class). Restarting a job that fails the same way forever is worse
      # than leaving it stopped: it hides the fault and burns the GPU.
      if [ "$train_restarts" -ge 3 ] && ! epoch_completed; then
        if [ "$loop_warned" -eq 0 ]; then
          log "GIVING UP on training: ${train_restarts} restarts with no epoch ever"
          log "  completing. This is a real fault, not a transient one."
          log "  Last error from $LOGT:"
          grep -E "Error|error:|Traceback|ValueError|RuntimeError" "$LOGT" \
            | tail -3 | sed 's/^/    /'
          loop_warned=1
        fi
        sleep "$CHECK_EVERY"; continue
      fi

      # PREFLIGHT. train_when_ready.sh validates the manifest before its first
      # launch; the supervisor used to skip that and start training directly,
      # which is exactly how a known-bad dataset reached the GPU after preflight
      # had already refused it.
      if ! ./scripts/preflight_manifest.sh lavdf >> "$LOGT" 2>&1; then
        if [ "$loop_warned" -eq 0 ]; then
          log "NOT starting training: manifest preflight failed (see $LOGT)"
          loop_warned=1
        fi
        sleep "$CHECK_EVERY"; continue
      fi

      restarts=$((restarts + 1))
      train_restarts=$((train_restarts + 1))
      log "RESTART #${restarts}: LAV-DF training (attempt ${train_restarts}, resumes from checkpoint)"
      nohup bash -c "./scripts/pod.sh exec '.venv/bin/python -u -m src.training.train --config=configs/lavdf.yaml'" \
        >> "$LOGT" 2>&1 &
      sleep 20
    fi
  fi

  # ----------------------------------------------------- 4. DFDC download
  # Independent of the GPU pipeline: head-node only, so it never waits on the pod.
  if ! download_done; then
    if ! running "download_dfdc"; then
      restarts=$((restarts + 1))
      log "RESTART #${restarts}: DFDC download (skips files already on disk)"
      nohup ./scripts/download_dfdc_when_clear.sh >> logs/download_dfdc.log 2>&1 &
      sleep 10
    fi
  fi

  sleep "$CHECK_EVERY"
done

log "STOP_SUPERVISOR found -- supervisor exiting. Jobs keep running."
rm -f "$STOP_FILE"
