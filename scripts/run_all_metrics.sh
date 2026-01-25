#!/usr/bin/env bash
# run_all_metrics.sh
# Simple wrapper to execute compute_metrics.py

FFMPEG_BIN="ffmpegF/ffmpeg"
REF_DIR="Metrics/Original"
TEST_BASE="Metrics"
OUT_BASE="Results/results"

# List of test folders
TEST_FOLDERS=(
  "L2A Cascading Trace"
  "Latency Cascading Trace"
  "LoL+ Cascading Trace"
  "L2A LTE Trace"
  "Latency LTE Trace"
  "LoL+ LTE Trace"
  "L2A Step Trace"
  "Latency Step Trace"
  "LoL+ Step Trace"
)

for test in "${TEST_FOLDERS[@]}"; do
  out_csv="${OUT_BASE}_$(echo "$test" | tr ' ' '_').csv"
  echo "Running metrics for: $test → $out_csv"
  python compute_metrics.py \
    --test "$TEST_BASE/$test" \
    --ref "$REF_DIR" \
    --ffmpeg "$FFMPEG_BIN" \
    --out_csv "$out_csv"
done
