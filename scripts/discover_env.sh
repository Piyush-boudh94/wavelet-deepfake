#!/usr/bin/env bash
# Re-run Step 0 environment discovery. Everything in docs/ENVIRONMENT.md came from this.
# Safe: read-only, no mutations.
set -uo pipefail
NS="${WMAMBA_NS:-dgx-s-bmu-cse-240577-restricted}"

hdr() { printf '\n===== %s =====\n' "$1"; }

hdr "HEAD NODE"
hostname; echo "cpu threads: $(nproc)"; free -h | head -2
echo -n "gpu on head node: "; command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "NONE (expected)"
echo -n "nvcc: "; (nvcc --version 2>/dev/null | tail -1) || echo "absent"
python3 --version

hdr "STORAGE"
df -h ~ | tail -1
echo "NOTE: / is shared with other users; check before large dataset transfers."

hdr "SCHEDULER"
echo -n "slurm: "; sinfo -s >/dev/null 2>&1 && echo "WORKING" || echo "present but NON-FUNCTIONAL (no config source) -> use kubernetes"
echo -n "kubectl context: "; kubectl config current-context 2>&1

hdr "NAMESPACE QUOTA ($NS)"
kubectl describe resourcequota -n "$NS" 2>&1 | sed -n '3,20p'

hdr "PODS"
kubectl get pods -n "$NS" -o wide 2>&1

hdr "GPU INSIDE POD"
if POD=$(kubectl get pods -n "$NS" -l app="${WMAMBA_APP:-wmamba-dev}" \
        --field-selector=status.phase=Running \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) && [ -n "$POD" ]; then
    kubectl exec -n "$NS" "$POD" -- bash -lc '
        nvidia-smi -L 2>/dev/null
        /home/dgx-s-bmu-cse-240577/research/.venv/bin/python - <<PY
import torch
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"{p.name}  {p.total_memory/1024**3:.1f} GiB usable  {p.multi_processor_count} SMs")
PY' 2>&1
else
    echo "no running pod; kubectl apply -f scripts/k8s/dev-pod.yaml"
fi
