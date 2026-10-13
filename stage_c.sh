#!/bin/zsh
# Stage C of PLAN_PRUNING.md for one fine-tuned finalist:
#   export fp32 ONNX -> official fp32 measurement -> static QDQ int8 with the PLAN.md winner
#   (per-channel, Percentile 99.99, S8S8, re-calibrated on the pruned model) -> official int8 measurement.
# Usage: ./stage_c.sh <finalist name in models/>  "<what we did>"
# Timing (timing.py) and the AI Hub submission are run separately in a clean session.
set -e
cd "$(dirname "$0")"
PY=edge-ai-env/bin/python
name=$1; what=$2
mkdir -p models/stageC
$PY export_onnx.py models/$name.pt -o models/stageC/${name}_fp32.onnx 2>&1 | grep -v -i warn
$PY verify_export.py models/$name.pt --onnx models/stageC/${name}_fp32.onnx --images 500 2>&1 | grep -v -i warn
$PY measure.py models/stageC/${name}_fp32.onnx --name ${name}_fp32 --what "$what; fp32 ONNX export" 2>&1 | grep -E "accuracy  |params|saved"
$PY quantize.py --input models/stageC/${name}_fp32.onnx --name ${name}_int8 --per-channel --calib-method percentile --percentile 99.99 --types s8s8 2>&1 | grep -v -i warn | grep -E "saved|PASS|FAIL|agree"
mv models/${name}_int8.onnx models/${name}_int8.json models/stageC/ 2>/dev/null || true
$PY measure.py models/stageC/${name}_int8.onnx --name ${name}_int8 --what "$what; then static QDQ int8 (per-channel, Percentile 99.99, S8S8, 300 calibration images, re-calibrated)" 2>&1 | grep -E "accuracy  |params|saved"
