#!/usr/bin/env bash
# Build multi-arch images (Intel/AMD + Apple Silicon) and push to Docker Hub.
#
# Uses the account from `docker login` when DOCKERHUB_USER is not set (reads
# Docker credential helper for registry index.docker.io).
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

PLATFORMS="linux/amd64,linux/arm64"
BUILDER_NAME="email-priority-multiarch"

detect_dockerhub_user() {
  if [[ -n "${DOCKERHUB_USER:-}" ]]; then
    printf '%s' "${DOCKERHUB_USER}"
    return 0
  fi

  local config="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
  if [[ ! -f "$config" ]]; then
    return 1
  fi

  local helper
  helper="$(python3 -c "import json; c=json.load(open('$config')); print(c.get('credsStore') or '')" 2>/dev/null || true)"
  if [[ -z "$helper" ]]; then
    return 1
  fi

  local helper_bin="docker-credential-${helper}"
  if ! command -v "$helper_bin" &>/dev/null; then
    return 1
  fi

  printf '{"ServerURL":"https://index.docker.io/v1/"}' \
 | "$helper_bin" get 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('Username') or '')" 2>/dev/null || true
}

setup_builder() {
  if docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
    docker buildx use "$BUILDER_NAME"
  else
    docker buildx create \
      --name "$BUILDER_NAME" \
      --driver docker-container \
      --use \
      --bootstrap
  fi
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
echo "== Platforms: ${PLATFORMS} =="

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
echo "== Done. Images are multi-arch (amd64 + arm64) on Docker Hub. =="
echo "Make each repository PUBLIC if others should pull without logging in:"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/email-priority-ml-api/settings"
echo "  https://hub.docker.com/repository/docker/${DOCKERHUB_USER}/email-priority-rails-app/settings"
echo "  → Change visibility to Public → Save."
echo ""
echo "On another machine (clone repo for ./ml volume mounts):"
echo "  export DOCKERHUB_USER=${DOCKERHUB_USER}"
echo "  docker compose -f docker-compose.hub.yml pull"
echo "  docker compose -f docker-compose.hub.yml up"
