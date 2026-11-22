#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manuscript_dir="$repo_dir/manuscript"

case "${1:-rebuild}" in
  rebuild)
    make -C "$manuscript_dir" rebuild
    ;;
  setup)
    make -C "$manuscript_dir" setup
    ;;
  clean)
    make -C "$manuscript_dir" clean
    ;;
  build)
    make -C "$manuscript_dir" build
    ;;
  figures)
    make -C "$manuscript_dir" figures
    ;;
  audit)
    make -C "$manuscript_dir" audit
    ;;
  *)
    echo "Usage: $0 [rebuild|setup|build|clean|figures|audit]" >&2
    exit 2
    ;;
esac
