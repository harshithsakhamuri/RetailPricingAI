#!/usr/bin/env sh
cd "$(dirname "$0")/.."
if command -v open >/dev/null 2>&1; then open docs/index.html; else xdg-open docs/index.html; fi
