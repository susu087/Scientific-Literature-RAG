#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/autodl-tmp/susu}"
BRANCH="${2:-main}"
REPO_URL="${3:-}"

if ! command -v git >/dev/null 2>&1; then
  echo "[sync_pull_autodl] git is not installed."
  exit 1
fi

if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
  echo "[sync_pull_autodl] Updated existing repo: $REPO_DIR ($BRANCH)"
  exit 0
fi

if [ -d "$REPO_DIR" ] && [ ! -d "$REPO_DIR/.git" ]; then
  echo "[sync_pull_autodl] Directory exists but is not a git repo: $REPO_DIR"
  exit 1
fi

if [ -z "$REPO_URL" ]; then
  echo "[sync_pull_autodl] Repo not found at $REPO_DIR."
  echo "Usage for first clone:"
  echo "  bash scripts/sync_pull_autodl.sh /root/autodl-tmp/susu main <repo-url>"
  exit 1
fi

mkdir -p "$(dirname "$REPO_DIR")"
git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
echo "[sync_pull_autodl] Cloned repo to: $REPO_DIR ($BRANCH)"
