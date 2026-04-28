#!/usr/bin/env bash
# Bootstrap script for a fresh Amazon Linux 2023 EC2 instance.
# Run as ec2-user. Idempotent — safe to re-run.

set -euo pipefail

echo "==> Updating system packages"
sudo dnf update -y

echo "==> Installing Docker"
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "==> Installing Docker Compose plugin"
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p "$DOCKER_CONFIG/cli-plugins"
COMPOSE_VERSION="v2.29.7"
curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
  -o "$DOCKER_CONFIG/cli-plugins/docker-compose"
chmod +x "$DOCKER_CONFIG/cli-plugins/docker-compose"

echo "==> Installing CloudWatch Agent"
sudo dnf install -y amazon-cloudwatch-agent || true

echo "==> Done. Log out and back in (or 'newgrp docker') so docker group takes effect."
echo "==> Then: cd fraud-detection-pipeline && docker compose up -d --build"
