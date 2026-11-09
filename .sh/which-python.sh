#!/usr/bin/env bash
# Find a runnable Python for this repo (prefers the project's tooling).
for c in python3.13 python3.12 python3 /usr/local/bin/python3 /opt/homebrew/bin/python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null; then
    echo "$c"
    exit 0
  fi
done
echo "NO_PYTHON_FOUND"
exit 1
