#!/usr/bin/env bash
# Recommended way to run the registration pipeline -- wraps
# `python -m registration_ants.pipeline` and captures ANTs' own per-iteration
# stdout output (which the C++ core writes directly to the process's stdout
# file descriptor, bypassing Python's logging -- see pipeline.py's module
# docstring) into the SAME run.log the pipeline's own step markers go to, so
# nothing is lost even for long/nohup'd runs.
#
# Usage:
#   ./run_pipeline.sh configs/s12t.yaml
#   nohup ./run_pipeline.sh configs/s12t.yaml > /dev/null 2>&1 &


if [ $# -ne 1 ]; then
    echo "Usage: $0 path/to/config.yaml" >&2
    exit 1
fi

CONFIG_PATH="$1"

# Resolve output_dir the same way the pipeline itself does (it depends on
# config.atlas.source selecting an atlas_variants entry), so run.log always
# lands next to the rest of that run's outputs.
OUTPUT_DIR=$(python -c "
import sys
from registration_ants.config import load_config
print(load_config(sys.argv[1])['output_dir'])
" "$CONFIG_PATH")

mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/run.log"

python -m registration_ants.pipeline "$CONFIG_PATH" 2>&1 | tee -a "$LOG_PATH"
