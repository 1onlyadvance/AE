#!/bin/bash

# This script finds all .cpp and .cu files and formats them for your CMakeLists.txt.

# --- REAL SOURCES ---
echo "# --- List of REAL Source Files ---"
echo "set(ARES_REAL_SOURCES"
# Find all .cpp/.cu files but exclude the build directory and stub files
find ares_edge_system -path "ares_edge_system/build" -prune -o \
-type f \( -name "*.cpp" -o -name "*.cu" \) -not -name "*_stub.cpp" -print \
| sort | while read -r source_file; do
  echo "    \"${source_file}\""
done
echo ")"

echo "" # Add a blank line for readability

# --- STUB SOURCES ---
echo "# --- List of STUB Source Files ---"
echo "set(ARES_STUB_SOURCES"
# Find only the files that DO end in _stub.cpp
find ares_edge_system -type f -name "*_stub.cpp" \
| sort | while read -r source_file; do
  echo "    \"${source_file}\""
done
echo ")"