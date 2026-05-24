#!/usr/bin/env bash
# Forward the kagent domain-agent A2A endpoints so the locally-running
# orchestrator-agent can delegate to them.
#
# Ports must match the *_AGENT_A2A_URL entries in .env:
#   IMS_AGENT_A2A_URL=http://localhost:9001/
#   JIRA_AGENT_A2A_URL=http://localhost:9002/
#
# Requires: VPN connected, kubectl context = surf-orchestration-dev-aks1.

set -euo pipefail

NAMESPACE="${NAMESPACE:-kagent}"
IMS_SERVICE="${IMS_SERVICE:-ims-mcp-agent}"
JIRA_SERVICE="${JIRA_SERVICE:-jira-mcp-agent}"
IMS_LOCAL_PORT="${IMS_LOCAL_PORT:-9001}"
JIRA_LOCAL_PORT="${JIRA_LOCAL_PORT:-9002}"
REMOTE_PORT="${REMOTE_PORT:-8080}"

pids=()

cleanup() {
    echo
    echo "Stopping port-forwards..."
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

forward() {
    local service="$1" local_port="$2"
    echo "  ${NAMESPACE}/${service}: 127.0.0.1:${local_port} -> svc:${REMOTE_PORT}"
    kubectl port-forward -n "${NAMESPACE}" "svc/${service}" "${local_port}:${REMOTE_PORT}" >/dev/null &
    pids+=("$!")
}

echo "Port-forwarding domain agents:"
forward "${IMS_SERVICE}"  "${IMS_LOCAL_PORT}"
forward "${JIRA_SERVICE}" "${JIRA_LOCAL_PORT}"
echo "(Ctrl-C to stop)"

wait
