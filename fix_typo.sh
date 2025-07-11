#!/bin/bash

# This script fixes the known typo in spectrum_waterfall_kernel.cu

TARGET_FILE="ares_edge_system/cew/kernels/spectrum_waterfall_kernel.cu"

if [ -f "$TARGET_FILE" ]; then
  echo "Found file: $TARGET_FILE"
  echo "Replacing 'LOG10_SCALE' with 'ares::cew::LOG_SCALE'..."
  sed -i 's/LOG10_SCALE/ares::cew::LOG_SCALE/g' "$TARGET_FILE"
  echo "Fix applied successfully."
else
  echo "Error: Could not find file $TARGET_FILE"
fi