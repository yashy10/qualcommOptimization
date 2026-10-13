# Standing rules for this project

- Read `plan.md` (quantization) and `PLAN_PRUNING.md` (pruning) before every task. They are
  the source of truth for this project; if they conflict with this file, this file wins.
- Accuracy is evaluated on all 27,000 EuroSAT images (judge instruction; split label
  `full_dataset_judge_instruction`), overall and per class.
- Baseline = `results/official_baseline_onnx_fp32.json`. Compare ONNX with ONNX.
- Export: opset 13, `dynamo=False`, input name `input`, shape (1, 3, 224, 224).
- Static QDQ quantization only. Never dynamic quantization.
- Always run `quant_pre_process` before quantizing.
- Calibration uses the fixed calibration-indices file only
  (`subsets/calibration_indices_300_seed42.json`).
- Never time models while other heavy jobs run. Timing is done only at milestones via
  `compare.py`. Rough timings must be labelled provisional.
- Every run is logged to the dashboard with a "What we did" description
  (`measure.py --what "..."`).
- Never overwrite saved results; never delete, only archive (`results/archive/`).
- Stop and report at the end of each step.
