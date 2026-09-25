#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$HERE/code/experiments/source_pretraining_and_transfer/included_type_adaptation_matrix"
cd "$EVALUATION_DIR"
python -u submit.py
