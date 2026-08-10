#!/usr/bin/env bash
# Resolve the current wmamba-dev pod and run a command inside it.
#
#   scripts/pod.sh name                 # print current pod name
#   scripts/pod.sh exec <cmd...>        # run a command in the pod
#   scripts/pod.sh py   <args...>       # run the project venv python
#   scripts/pod.sh pip  <args...>       # run the project venv pip
#   scripts/pod.sh shell                # interactive shell
#
# Pod names change every time the Deployment reschedules -- never hardcode them.
set -euo pipefail

NS="${WMAMBA_NS:-dgx-s-bmu-cse-240577-restricted}"
APP="${WMAMBA_APP:-wmamba-dev}"
ROOT="/home/dgx-s-bmu-cse-240577/research"

pod_name() {
    local p
    p=$(kubectl get pods -n "$NS" -l "app=$APP" \
        --field-selector=status.phase=Running \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    [ -n "$p" ] || { echo "No running '$APP' pod in $NS. Try: kubectl apply -f scripts/k8s/dev-pod.yaml" >&2; exit 1; }
    echo "$p"
}

cmd="${1:-shell}"; shift || true
case "$cmd" in
    name)  pod_name ;;
    exec)  kubectl exec -n "$NS" "$(pod_name)" -- bash -lc "cd $ROOT && $*" ;;
    py)    kubectl exec -n "$NS" "$(pod_name)" -- bash -lc "cd $ROOT && .venv/bin/python $*" ;;
    pip)   kubectl exec -n "$NS" "$(pod_name)" -- bash -lc "cd $ROOT && .venv/bin/pip $*" ;;
    shell) kubectl exec -it -n "$NS" "$(pod_name)" -- bash ;;
    *)     echo "usage: pod.sh {name|exec|py|pip|shell} [args...]" >&2; exit 1 ;;
esac
