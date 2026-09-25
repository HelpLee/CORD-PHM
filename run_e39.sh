#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E39="$HERE/workspace/code/experiments/A_main/suite/39_complete_adaptation_matrix"
cd "$E39"
python -u submit.py
