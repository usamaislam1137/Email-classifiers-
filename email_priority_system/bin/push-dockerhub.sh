#!/usr/bin/env bash
# Build multi-arch images (Intel/AMD + Apple Silicon) and push to Docker Hub.
#
# Uses the account from `docker login` when DOCKERHUB_USER is not set (reads
# credHelpers / credsStore and Docker Hub URLs the way Docker Desktop stores them).
#
# Prerequisites:
#   docker login
#   Docker Buildx (included with Docker Desktop)
#
# Optional override:
#   export DOCKERHUB_USER=myhublogin
#   ./bin/push-dockerhub.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Comma-separated. Multi-arch (two platforms) is heavy; if export fails with I/O errors,
# try:  PUSH_PLATFORMS=linux/arm64  (or linux/amd64) on one machine, push again on another arch if needed
PLATFORMS="${PUSH_PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER_NAME="email-priority-multiarch"
# docker-container driver uses a separate BuildKit container; its overlay2 / containerd
# can hit I/O errors. Default: "docker". Override: export BUILDX_DRIVER=docker-container
BUILDX_DRIVER="${BUILDX_DRIVER:-docker}"

detect_dockerhub_user() {
  if [[ -n "${DOCKERHUB_USER:-}" ]]; then
    printf '%s' "${DOCKERHUB_USER}"
    return 0
  fi

  local config="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
  if [[ ! -f "$config" ]]; then
    return 1
  fi

  # Docker Desktop often uses per-registry credHelpers (e.g. docker.io) instead of credsStore.
  # Try several ServerURL strings because helpers key off the URL used at login.
  CONFIG_PATH="$config" python3 <<'PY'
import base64
import json
import os
import shutil
import subprocess
import sys

def helper_get(helper_name: str, server_url: str):
    exe = shutil.which(f"docker-credential-{helper_name}")
    if not exe:
        return None
    payload = json.dumps({"ServerURL": server_url})
    try:
        p = subprocess.run(
            [exe, "get"],
            input=payload.encode(),
            capture_output=True,
            timeout=15,
        )
        if p.returncode != 0:
            return None
        data = json.loads(p.stdout.decode())
        u = (data.get("Username") or "").strip()
        return u or None
    except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None

def try_helper(helper_name: str):
    urls = [
        "https://index.docker.io/v1/",
        "https://index.docker.io/v2/",
        "registry-1.docker.io",
        "https://registry-1.docker.io/v2/",
        "docker.io",
        "https://docker.io/v1/",
    ]
    for url in urls:
        u = helper_get(helper_name, url)
        if u:
            return u
    return None

path = os.environ.get("CONFIG_PATH", "")
if not path or not os.path.isfile(path):
    sys.exit(1)

with open(path, encoding="utf-8") as f:
    cfg = json.load(f)

# 1) credHelpers: registry host -> helper (Docker Desktop typical)
for reg, helper in (cfg.get("credHelpers") or {}).items():
    r = reg.lower()
    if "docker.io" not in r and "docker.com" not in r and "index.docker.io" not in r:
        continue
    u = try_helper(helper)
    if u:
        print(u)
        sys.exit(0)
    # Also ask the helper using the config key itself as ServerURL
    u = helper_get(helper, reg if reg.startswith("http") else f"https://{reg}/v1/")
    if u:
        print(u)
        sys.exit(0)

# 2) Global credsStore
store = (cfg.get("credsStore") or "").strip()
if store:
    u = try_helper(store)
    if u:
        print(u)
        sys.exit(0)

# 3) Inline auth (older / CI)
for url, entry in (cfg.get("auths") or {}).items():
    if "docker.io" not in url.lower():
        continue
    auth = entry.get("auth")
    if not auth:
        continue
    try:
        raw = base64.b64decode(auth).decode("utf-8")
        if ":" in raw:
            print(raw.split(":", 1)[0])
            sys.exit(0)
    except Exception:
        pass

sys.exit(1)
PY
}

setup_builder() {
  if [[ "${BUILDX_PRUNE:-}" == "1" ]]; then
    echo "== buildx prune (clears bad BuildKit cache; helps input/output on export) ==" >&2
    docker buildx prune -af 2>/dev/null || true
  fi

  if [[ "$BUILDX_DRIVER" == "docker" ]]; then
    # Docker only allows ONE buildx instance with the "docker" driver. Creating
    # a second (e.g. "email-priority-multiarch") fails with:
    #   additional instances of driver "docker" cannot be created
    # So we always use the default builder for driver=docker; drop a stale named one.
    if docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
      echo "== Removing named builder $BUILDER_NAME (only one 'docker' driver instance allowed) ==" >&2
      docker buildx rm -f "$BUILDER_NAME" 2>/dev/null || true
    fi
    # buildx "default" builder is bound to the default *context*; otherwise:
    #   ERROR: run `docker context use default` to switch to default context
    local dctx
    dctx="$(docker context show 2>/dev/null || true)"
    if [[ -z "$dctx" || "$dctx" != "default" ]]; then
      echo "== docker context use default (was: ${dctx:-unknown}) ==" >&2
      docker context use default
    fi
    echo "== Using buildx 'default' (docker driver) ==" >&2
    docker buildx use default
    docker buildx inspect default --bootstrap &>/dev/null || true
    return
  fi

  if docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
    local cur
    cur="$(docker buildx inspect "$BUILDER_NAME" -f '{{.Driver}}' 2>/dev/null | tr -d '\r\n' || true)"
    if [[ -z "$cur" || "$cur" != "$BUILDX_DRIVER" ]]; then
      echo "== Recreating buildx (was driver: ${cur:-?}, want: $BUILDX_DRIVER) ==" >&2
      docker buildx rm -f "$BUILDER_NAME" 2>/dev/null || true
    else
      docker buildx use "$BUILDER_NAME"
      return
    fi
  fi
  docker buildx create \
    --name "$BUILDER_NAME" \
    --driver "$BUILDX_DRIVER" \
    --use \
    --bootstrap
}

HUB_USER="$(detect_dockerhub_user || true)"
if [[ -z "${HUB_USER}" ]]; then
  echo "Could not read Docker Hub username from your login." >&2
  echo "Run: docker login" >&2
  echo "Or set: export DOCKERHUB_USER=your_dockerhub_username" >&2
  exit 1
fi

export DOCKERHUB_USER="$HUB_USER"
ML_IMAGE="${DOCKERHUB_USER}/email-priority-ml-api:latest"
RAILS_IMAGE="${DOCKERHUB_USER}/email-priority-rails-app:latest"

echo "== Docker Hub user: ${DOCKERHUB_USER} (from login or DOCKERHUB_USER) =="
if [[ "$BUILDX_DRIVER" == "docker" ]]; then
  echo "== Platforms: ${PLATFORMS}  |  buildx: default (only one 'docker' driver; see script comments) =="
else
  echo "== Platforms: ${PLATFORMS}  |  buildx driver: ${BUILDX_DRIVER} (builder: ${BUILDER_NAME}) =="
fi
echo "   (from repo dir that contains this script: ${ROOT})"

setup_builder

echo "== Build & push ${ML_IMAGE} =="
docker buildx build \
  --platform "${PLATFORMS}" \
  -f Dockerfile.ml \
  -t "${ML_IMAGE}" \
  --push \
  --provenance=false \
  .

echo "== Build & push ${RAILS_IMAGE} =="
docker buildx build \
  --platform "${PLATFORMS}" \
  -f Dockerfile.rails \
  -t "${RAILS_IMAGE}" \
  --push \
  --provenance=false \
  .

echo ""
if [[ "$PLATFORMS" == *","* ]]; then
  echo "== Done. Multi-platform manifest pushed. =="
else
  echo "== Done. Single-platform image pushed (${PLATFORMS}). =="
fi
echo "Make each repository PUBLIC if others should pull without logging in:"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/email-priority-ml-api/settings"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/email-priority-rails-app/settings"
echo "  → Change visibility to Public → Save."
echo ""
echo "On another machine (clone repo for ./ml volume mounts):"
echo "  export DOCKERHUB_USER=${DOCKERHUB_USER}"
echo "  docker compose -f docker-compose.hub.yml pull"
echo "  docker compose -f docker-compose.hub.yml up"
