#!/usr/bin/env bash
# Self-healing live gateway for the Śabda-ALM app (admin-run, on the DGX Spark).
# Keeps the pranava-serve container + a cloudflared tunnel up, and re-registers the tunnel URL into
# Supabase `runtime_config.alm_gateway_url` (public-read) whenever it changes — so the Vercel app
# (auth-gated: admin/guest = live, user = replay) always finds the backend. Mirrors prabodha's
# scripts/ops/gateway_tunnel_service.sh, targeting Supabase instead of Vercel env.
#
# Run (admin, on the DGX):
#   SUPABASE_URL=https://<ref>.supabase.co SUPABASE_SERVICE_KEY=<service_role_key> \
#     nohup bash scripts/ops/alm_tunnel_service.sh > /tmp/alm_tunnel.log 2>&1 &
#   # reboot survival: install as a systemd --user service + `loginctl enable-linger`.
set -u

CF="${CF:-$HOME/.local/bin/cloudflared}"
PORT="${PORT:-8088}"
LOCAL="http://localhost:${PORT}"
SUPABASE_URL="${SUPABASE_URL:?set SUPABASE_URL}"
SERVICE_KEY="${SUPABASE_SERVICE_KEY:?set SUPABASE_SERVICE_KEY}"
IMAGE="${ALM_IMAGE:-prabhasa/nemo-gb10:26.02}"
PROJECTS="${PROJECTS:-/home/sharaths/projects}"
LOG="/tmp/alm_cf_tunnel.log"
URL_FILE="/tmp/alm_gateway_url.txt"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

ensure_server(){
  if ! curl -s -o /dev/null --max-time 8 "$LOCAL/health"; then
    log "server down -> (re)start pranava-serve"
    docker rm -f pranava-serve >/dev/null 2>&1
    docker run -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --name pranava-serve -p ${PORT}:${PORT} \
      -v ${PROJECTS}:/work -w /work \
      -e PYTHONPATH=/work/pranava/src:/work/prabhasa-samskrutam/src \
      -e HF_HOME=/work/pranava/.hf_cache \
      "$IMAGE" uvicorn pranava.serve.server:app --host 0.0.0.0 --port ${PORT} >/dev/null 2>&1
    for _ in $(seq 1 40); do curl -s -o /dev/null --max-time 5 "$LOCAL/health" && break; sleep 6; done
  fi
}

register_url(){  # $1 = new tunnel url → Supabase runtime_config.alm_gateway_url
  log "registering $1 into Supabase runtime_config"
  curl -s -X POST "${SUPABASE_URL}/rest/v1/runtime_config?on_conflict=key" \
    -H "apikey: ${SERVICE_KEY}" -H "Authorization: Bearer ${SERVICE_KEY}" \
    -H "Content-Type: application/json" -H "Prefer: resolution=merge-duplicates" \
    -d "[{\"key\":\"alm_gateway_url\",\"value\":\"$1\"}]" -o /dev/null
  echo "$1" > "$URL_FILE"
}

start_tunnel(){
  pkill -f "cloudflared tunnel --url $LOCAL" 2>/dev/null
  : > "$LOG"
  nohup "$CF" tunnel --url "$LOCAL" --no-autoupdate >"$LOG" 2>&1 &
  echo $! > /tmp/alm_cf_tunnel.pid
  local url=""
  for _ in $(seq 1 30); do
    url=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG" | head -1)
    [ -n "$url" ] && break; sleep 2
  done
  echo "${url:-}"
}

log "ALM tunnel service starting (port ${PORT})"
last=""
while true; do
  ensure_server
  # (re)establish the tunnel if the process died
  if ! kill -0 "$(cat /tmp/alm_cf_tunnel.pid 2>/dev/null)" 2>/dev/null; then
    url=$(start_tunnel)
    if [ -n "$url" ] && [ "$url" != "$last" ]; then register_url "$url"; last="$url"; log "live at $url"; fi
  fi
  sleep 30
done
