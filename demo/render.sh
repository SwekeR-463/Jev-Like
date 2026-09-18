#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv run --python 3.12 python -m demo.render
