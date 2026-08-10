#!/usr/bin/env bash
# Refuse to train on a dataset that cannot produce a metric.
#
# Exists because on 2026-08-04 training ran 55 times against a manifest whose
# val split held only real videos. Every epoch 0 finished, then video_auc raised
# "single class" and the process died -- six hours of GPU for nothing. The
# handoff script HAD caught it; the supervisor launched training anyway. Both
# now call this, so there is one check and no way around it.
#
#   ./scripts/preflight_manifest.sh lavdf   -> exit 0 pass, 1 fail
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DS=${1:?usage: preflight_manifest.sh <dataset>}

./scripts/pod.sh exec ".venv/bin/python - <<'PY'
import json, sys, collections
from pathlib import Path

mf = Path('data/processed/${DS}/manifest.json')
if not mf.exists():
    print(f'PREFLIGHT FAIL: {mf} missing'); sys.exit(1)
try:
    entries = json.loads(mf.read_text())['entries']
except Exception as e:
    print(f'PREFLIGHT FAIL: {mf} unreadable: {e}'); sys.exit(1)

by = collections.Counter((e['split'], e['label']) for e in entries)
bad = []
print(f'preflight {mf}: {len(entries)} entries')
for s in ('train', 'val', 'test'):
    r, f = by[(s, 0)], by[(s, 1)]
    print(f'  {s:5s} real={r:6d} fake={f:6d}')
    if s in ('train', 'val') and (r == 0 or f == 0):
        # train needs both to learn; val needs both or AUC is undefined and the
        # early-stopping probe crashes at the end of every epoch.
        bad.append(f'{s} has a single class (real={r}, fake={f})')
if not entries:
    bad.append('manifest is empty')

if bad:
    print('PREFLIGHT FAIL:')
    for b in bad:
        print('  -', b)
    sys.exit(1)
print('PREFLIGHT PASS')
PY"
