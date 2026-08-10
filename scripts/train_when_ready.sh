#!/usr/bin/env bash
# Hand off from LAV-DF preprocessing straight into training, with preflight.
#
# Purpose is purely to remove dead time: preprocessing finishes at some
# unpredictable hour and the GPU would otherwise sit idle until a human noticed.
#
# It will NOT start training blindly. Between the two phases it verifies the
# manifest is complete and both classes are present in every split, then runs a
# 2-step smoke test. Any failure aborts and leaves a clear reason in the log --
# far cheaper than discovering a bad manifest 20 hours into a run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CFG=configs/lavdf.yaml
LOG=logs/train_lavdf.log
EXPECT=30000

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------- 1. wait it out
log "waiting for LAV-DF preprocessing (all shards) to finish..."
while pgrep -f "preprocess.py --dataset lavdf" >/dev/null \
   || pgrep -f run_preprocess_shards >/dev/null; do
  sleep 120
done
log "preprocessing process has exited"

if ! grep -q "^merged .* into .*manifest.json" logs/preprocess_lavdf.log; then
  log "ABORT: preprocessing exited without writing a final manifest."
  log "       it was probably killed (pod wipe?). Re-run the preprocess command;"
  log "       it resumes, then re-run this script."
  exit 1
fi

# ------------------------------------------------------------ 2. preflight
# Shared with supervisor.sh so there is exactly ONE definition of "trainable".
log "preflight: validating manifest"
if ! ./scripts/preflight_manifest.sh lavdf 2>&1 | tee -a "$LOG"; then
  log "ABORT: manifest preflight failed -- not burning GPU on a bad dataset"
  exit 1
fi

# --------------------------------------------- 2b. ML-correctness audit
# Structural validity (both classes present) is not enough: this also checks
# split disjointness at video AND frame level, label sanity, that eval is
# deterministic/un-augmented while train is not, and that the val split can
# actually yield a defined AUC. A flawed dataset must never reach the GPU --
# on 2026-08-04 one did, and cost 7.8 h.
log "preflight: ML-correctness audit"
if ! ./scripts/pod.sh exec ".venv/bin/python scripts/audit_ml_correctness.py lavdf" \
     >> "$LOG" 2>&1; then
  log "ABORT: ML-correctness audit failed -- see $LOG"
  exit 1
fi
log "ML-correctness audit passed"

# ------------------------------------------------------ 3. two-step smoke test
log "preflight: 2-step smoke test"
if ! ./scripts/pod.sh exec ".venv/bin/python -m src.training.train --config=$CFG --smoke-steps=2" \
     >> "$LOG" 2>&1; then
  log "ABORT: smoke test failed -- see $LOG"
  exit 1
fi
log "smoke test passed"

# ----------------------------------------------------------- 4. full training
log "STARTING FULL TRAINING ($CFG)"
exec ./scripts/pod.sh exec ".venv/bin/python -u -m src.training.train --config=$CFG" \
     >> "$LOG" 2>&1
