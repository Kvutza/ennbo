#!/usr/bin/env bash
set -euo pipefail

# Run from repo root so paths from git status are correct
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Scanning for untracked or modified Python files (git status)..."

status_output="$(git status --porcelain --untracked-files=all)"

if [[ -z "$status_output" ]]; then
  echo "Working tree clean. No forgotten .py files."
  exit 0
fi

py_files="$(printf "%s\n" "$status_output" | awk '{print $2}' | grep -E '\.py$' || true)"

if [[ -z "$py_files" ]]; then
  echo "No forgotten .py files detected."
  exit 0
fi

echo "Potential forgotten Python files:"
printf "%s\n" "$py_files"
