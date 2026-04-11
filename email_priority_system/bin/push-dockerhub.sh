#!/usr/bin/env bash
# Build locally, tag, and push ML + Rails images to Docker Hub.
#
# Prerequisites:
#   1. Create a Docker Hub account (https://hub.docker.com)
#   2. docker login (must match DOCKERHUB_USER below)
#
# Usage:
#   export DOCKERHUB_USER=your_dockerhub_username
#   ./bin/push-dockerhub.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [[ -z "${DOCKERHUB_USER:-}" ]]; then
  echo "Set DOCKERHUB_USER to your Docker Hub username (the one you used with 'docker login')." >&2
  echo "Example: export DOCKERHUB_USER=myname && ./bin/push-dockerhub.sh" >&2
  exit 1
fi

ML_IMAGE="${DOCKERHUB_USER}/email-priority-ml-api:latest"
RAILS_IMAGE="${DOCKERHUB_USER}/email-priority-rails-app:latest"

echo "== Building (project: $(basename "${ROOT}")) =="
docker compose build

# Compose v2 default project name is the directory name (e.g. email_priority_system-ml-api)
LOCAL_ML_NAME="$(basename "${ROOT}")-ml-api:latest"
LOCAL_RAILS_NAME="$(basename "${ROOT}")-rails-app:latest"

echo "== Tagging =="
docker tag "${LOCAL_ML_NAME}" "${ML_IMAGE}"
docker tag "${LOCAL_RAILS_NAME}" "${RAILS_IMAGE}"

echo "== Pushing ${ML_IMAGE} =="
docker push "${ML_IMAGE}"

echo "== Pushing ${RAILS_IMAGE} =="
docker push "${RAILS_IMAGE}"

echo "Done. On another machine (with this repo cloned):"
echo "  export DOCKERHUB_USER=${DOCKERHUB_USER}"
echo "  docker compose -f docker-compose.hub.yml pull"
echo "  docker compose -f docker-compose.hub.yml up"
