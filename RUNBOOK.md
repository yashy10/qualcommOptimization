# Runbook: reproduce the submitted model from the provided checkpoint

Use a fresh workspace to reproduce the pipeline without overwriting saved results.
Everything runs from this folder with the supplied `student_mobilenetv2_distilled.pt`
and `teacher_resnet50_eurosat.pt` placed next to the scripts. CPU only is enough;
`--device mps` (Apple GPU) or `--device cuda` selects an accelerator for recovery
and its subset evaluations. The recorded selected run used MPS.

```bash
# 0. Environment (Python 3.10+)
python3.11 -m venv edge-ai-env && source edge-ai-env/bin/activate
pip install -r requirements.txt
# EuroSAT: torchvision downloads it into ./data on first use (or copy a cached ./data folder)

# 1. Baseline: export fp32 ONNX and measure it on all 27,000 images
python export_onnx.py                                  # -> models/student_fp32.onnx
python measure.py models/student_fp32.onnx --name official_baseline_onnx_fp32

# 2. Prune: rebuild the depth-pruned model exactly from its config
#    (residual blocks 6, 9, 16 -> Identity, adaptive BN, seed 0)
python prune.py --config configs/a2_blocks3.json       # -> models/a2_blocks3.pt

# 3. Teacher logits for distillation (one-off, cached)
python finetune.py --cache-teacher                     # -> cache/teacher_logits_ft20.pt

# 4. Recovery fine-tune: 3 epochs, KD, lr 0.001, batch 32, seed 0
python finetune.py models/a2_blocks3.pt --name b3_kd_lr001_e3 --epochs 3 --lr 0.001 --device cpu
#    (--device mps on Apple Silicon: ~30 s/epoch instead of ~16 min)

# 5. Export, quantize (static QDQ, per-channel, Percentile 99.99, S8S8, 300 fixed images),
#    sanity-check, and measure fp32 and int8 on all 27,000 images
./stage_c.sh b3_kd_lr001_e3 "blocks 6,9,16 removed + 3-epoch KD recovery"
#    -> models/stageC/b3_kd_lr001_e3_int8.onnx  (this is student_model_int8.onnx)

# 6. Compare everything against the baseline, and open the dashboard
python compare.py
open results/dashboard.html
```

## Verifying the submitted file directly (no training)

```bash
python check_quant.py submission/student_model_int8.onnx           # loads, static QDQ, no dynamic ops, size ratio
python measure.py submission/student_model_int8.onnx --name submitted_int8
```

The direct-verification commands above run from the repository root after the
baseline export. From packaged `submission/code/`, use `../student_model_int8.onnx`
instead. For repeated clean timing, run `python timing.py
official_baseline_onnx_fp32 submitted_int8 --rounds 5` separately on mains power
with low background load.

`measure.py` prints accuracy (overall and worst class), params, MACs, size, latency,
peak RAM and energy, and writes `results/submitted_int8.json`; `compare.py` shows the
ratios against the baseline.

## Where the numbers in metrics.json come from

- Accuracy, per-class, params, MACs, size: `results/b3_kd_lr001_e3_int8.json` (all 27,000 images).
- Selected-model latency, RAM, energy: provisional single-session values from
  `results/b3_kd_lr001_e3_int8.json`, measured under background CPU load.
  Baseline timing uses the earlier five-round milestone in `results/timing/`.
  Their local ratios are not a verified matched-session benchmark.
- AI Hub: `results/ai_hub_jobs.json` and `results/aihub_profiles/` (Dragonwing RB3 Gen 2 Vision Kit).
- Every pruned model's exact recipe: `configs/<name>.json`; every quantized model's settings: `models/stageC/<name>_int8.json`.

Notes: recovery and pruning use seed 0; calibration uses seed 42. GPU vs CPU
training can produce different floating-point results and accuracy; exact numerical
or byte-for-byte reproduction across devices is not guaranteed.
Timing and energy depend on the machine, so they are scored as ratios against the
baseline measured on the same machine.
