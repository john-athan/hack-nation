#!/usr/bin/env bash
# One-command local launcher / health-check for Genome Firewall — for a fresh clone or the
# backup scenario (box up, network flaky). It starts Streamlit detached (survives this shell via
# setsid), waits for the health endpoint to go green, verifies the curated demo genomes' FASTAs
# are cached, then prints the URL(s) + PID and EXITS 0 — leaving Streamlit running in the
# background. Stop it later with:  pkill -f "streamlit run demo/app.py"   (or kill the printed PID).
#
# Usage:  ./demo.sh            # port 8501
#         PORT=8600 ./demo.sh  # override port
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8501}"
HEALTH_URL="http://localhost:${PORT}/_stcore/health"
LOCAL_URL="http://localhost:${PORT}"
HEALTH_TIMEOUT=60
LOG="$(mktemp -t genome-firewall-demo.XXXXXX.log)"

# Best-effort: make the OpenAI key available for the live rationale layer. Absence is fine —
# the demo falls back to the pre-baked cache / deterministic templates, so never fail here.
if [[ -f "$HOME/.hack.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HOME/.hack.env"
  set +a
fi

echo "🧬 Genome Firewall — starting Streamlit on port ${PORT}…"

# Detach with setsid so the server survives this launcher exiting (and any cycle teardown).
setsid uv run --extra demo streamlit run demo/app.py \
  --server.port "$PORT" --server.headless true >"$LOG" 2>&1 </dev/null &
SESSION_PID=$!

# Poll the Streamlit health endpoint until it returns HTTP 200 (or we give up).
healthy=0
for ((i = 0; i < HEALTH_TIMEOUT; i++)); do
  if ! kill -0 "$SESSION_PID" 2>/dev/null; then
    echo "❌ Streamlit process exited before becoming healthy. Last log lines:" >&2
    tail -n 20 "$LOG" >&2 || true
    exit 1
  fi
  code="$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH_URL" 2>/dev/null || true)"
  if [[ "$code" == "200" ]]; then
    healthy=1
    break
  fi
  sleep 1
done

if [[ "$healthy" -ne 1 ]]; then
  echo "❌ Streamlit did not become healthy within ${HEALTH_TIMEOUT}s. Log: $LOG" >&2
  tail -n 20 "$LOG" >&2 || true
  exit 1
fi

# Confirm the curated demo genomes' cached FASTAs are present — the instant, offline demo path.
echo "Checking curated demo genome FASTAs…"
ids="$(uv run python -c "
import json
from pathlib import Path
data = json.loads(Path('data/demo_genomes.json').read_text())
print('\n'.join(g['id'] for g in data['genomes']))
" 2>/dev/null || true)"
if [[ -z "$ids" ]]; then
  echo "  ⚠️  Could not read data/demo_genomes.json — skipping FASTA check."
else
  while IFS= read -r id; do
    [[ -z "$id" ]] && continue
    if [[ -f "data/fasta/${id}.fna" ]]; then
      echo "  ✅ data/fasta/${id}.fna"
    else
      echo "  ⚠️  MISSING data/fasta/${id}.fna"
    fi
  done <<<"$ids"
fi

echo
echo "════════════════════════════════════════════════════════════"
echo "✅ Genome Firewall is LIVE"
echo "   Local URL : ${LOCAL_URL}"
if [[ -s docs/DEMO_URL.txt ]]; then
  echo "   Public URL: $(head -n1 docs/DEMO_URL.txt)"
fi
echo "   PID       : ${SESSION_PID}   (log: ${LOG})"
echo "   Stop with : pkill -f \"streamlit run demo/app.py\""
echo "════════════════════════════════════════════════════════════"
