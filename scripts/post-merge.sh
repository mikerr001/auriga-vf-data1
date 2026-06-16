#!/bin/bash
set -e

cd auriga-virtual-fiducial-analyzer
uv pip install -r requirements.txt --quiet 2>/dev/null || pip install -r requirements.txt --quiet
