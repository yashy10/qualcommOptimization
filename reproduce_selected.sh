#!/bin/sh
# Reproduce the selected B3 model in a fresh copy of the source workspace.
# Required: supplied student/teacher .pt checkpoints and data/eurosat/2750.
# Original recovery used MPS. Set TRAIN_DEVICE=cpu for a CPU-only environment.
set -eu
cd "$(dirname "$0")"
PYTHON_BIN=${PYTHON_BIN:-python3}
TRAIN_DEVICE=${TRAIN_DEVICE:-mps}
run_name=b3_kd_lr001_e3
for checkpoint in student_mobilenetv2_distilled.pt teacher_resnet50_eurosat.pt; do
    test -f "$checkpoint" || { echo "Missing supplied checkpoint: $checkpoint" >&2; exit 1; }
done
for artifact in models/a2_blocks3.pt models/b3_kd_lr001_e3.pt models/stageC/b3_kd_lr001_e3_fp32.onnx models/stageC/b3_kd_lr001_e3_int8.onnx models/b3_kd_lr001_e3_int8.onnx; do
    test ! -e "$artifact" || { echo "Use a fresh workspace; existing artifact: $artifact" >&2; exit 1; }
done
mkdir -p models/stageC cache
if test ! -f models/student_fp32.onnx; then
    "$PYTHON_BIN" export_onnx.py
fi
"$PYTHON_BIN" prune.py --config configs/a2_blocks3.json
"$PYTHON_BIN" finetune.py --cache-teacher
"$PYTHON_BIN" finetune.py models/a2_blocks3.pt --name "$run_name" --epochs 3 --lr 0.001 --weight-decay 0.00004 --batch-size 32 --alpha 0.7 --temperature 4 --device "$TRAIN_DEVICE"
"$PYTHON_BIN" export_onnx.py "models/$run_name.pt" -o "models/stageC/${run_name}_fp32.onnx"
"$PYTHON_BIN" verify_export.py "models/$run_name.pt" --onnx "models/stageC/${run_name}_fp32.onnx" --images 500
"$PYTHON_BIN" quantize.py --input "models/stageC/${run_name}_fp32.onnx" --name "${run_name}_int8" --per-channel --calib-method percentile --percentile 99.99 --types s8s8
mv "models/${run_name}_int8.onnx" "models/${run_name}_int8.json" models/stageC/
echo "Selected model rebuilt: models/stageC/${run_name}_int8.onnx"
echo "Evaluate with measure.py; use timing.py separately under quiet mains-power conditions."
