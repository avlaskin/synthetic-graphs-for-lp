#!/bin/bash

# This is graph set index i 0, 1, ...
export GRAPH_SIMBOL="0"
# Graph index in the set j in [0; 9]
export TASK_INDEX="0"
# Graph prefix faab, gaaa, gaab
export TASK_PREFIX="faab"

# Ensure local run has empty project:
export PROJECT_ID=""

cd ./jobs
# This will only work for setup env.
# SBM will not work locally
python main.py
