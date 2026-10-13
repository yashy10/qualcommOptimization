# Minimalist Edge AI Challenge — Technical Report

**Submitted model:** `student_model_int8.onnx` — the provided MobileNetV2 student, statically quantized to int8 (QDQ, per-channel, Percentile-99.99 calibration). Accepts `(1, 3, 224, 224)` float32, input name `input`.

## 1. Baseline (official, all 27,000 EuroSAT images, judge instruction)

fp32 ONNX export of `student_mobilenetv2_distilled.pt` (opset 13, `dynamo=False`), measured on an Apple M3 Pro, CPU only, ONNX Runtime 1.31: **99.16 %** accuracy, worst class PermanentCrop 95.84 %, 2.220 M params, 0.299 G MACs, 8.51 MB, 2.55 ms, 87.9 MB peak RAM, 0.0599 J/inference (CodeCarbon TDP estimate). PyTorch reference: 19.9 ms, 303 MB.

## 2. Final model vs baseline (same machine, same scripts)

| | Baseline fp32 | **Final int8 (Q3a)** | Ratio |
|---|---|---|---|
| Accuracy | 99.16 % | **98.92 %** | 99.76 % retained |
| Worst class | PermanentCrop 95.84 % | PermanentCrop 95.24 % | no class drops > 1 pt |
| Params / MACs | 2.220 M / 0.299 G | 2.220 M / 0.299 G | unchanged (quantization) |
| File size | 8.51 MB | **2.49 MB** | **3.41x smaller** |
| Latency (median, 5 clean rounds) | 2.55 ms | **0.84 ms** | **3.06x faster** |
| Peak RAM | 87.9 MB | 76.9 MB | 1.14x less |
| Energy / inference (TDP estimate) | 0.0599 J | **0.0194 J** | **3.09x less** |
| Qualcomm AI Hub, RB3 Gen 2 (QCS6490), CPU | 26.7 ms, 29.5 MB | **6.2 ms**, 29.4 MB | **4.3x faster** |

AI Hub device: the Arduino UNO Q's QRB2210 is not in the catalog; the Dragonwing RB3 Gen 2 Vision Kit is the lowest-tier Qualcomm Linux IoT device listed. Both models ran with every layer on CPU, so the comparison is like-for-like (jobs `jp8j1yxo5`/`jgk68x4n5` and `jgd6216zp`/`j5wy46y4g`).

## 3. Quantization: what we tried, kept and discarded

All runs: `quant_pre_process` → static `quantize_static` (QDQ) → 300 fixed, class-balanced calibration images → accuracy on all 27,000 images. Targets: ≥ 98.9 % accuracy, PermanentCrop ≥ 95 %, ≤ 2.5 MB.

| Run | Setting | Accuracy | PermanentCrop | Size | Verdict |
|---|---|---|---|---|---|
| Q0 | per-tensor, MinMax, U8S8 | 96.32 % | 93.44 % | 2.28 MB | negative result: per-tensor hurts depthwise layers (Forest −11.7 pts) |
| Q1 | per-channel, MinMax, S8S8 | 98.09 % | 96.96 % | 2.49 MB | per-channel recovers 1.8 pts |
| Q1b | Q1 with U8S8 | 98.09 % | 96.96 % | 2.49 MB | identical predictions: data type irrelevant |
| Q2 | per-channel, Entropy | 98.09 % | 96.96 % | 2.49 MB | bit-identical scales to MinMax: KL chose full range on every ReLU6-bounded tensor |
| **Q3a** | **per-channel, Percentile 99.99** | **98.92 %** | **95.24 %** | **2.49 MB** | **winner: clipping the 0.01 % tail recovers 0.83 pts** |
| Q3b | Percentile 99.999 | 98.69 % | 95.92 % | 2.49 MB | over-clips |
| Q4 | Q3a + first conv & classifier fp32 | 98.86 % | 94.88 % | 2.53 MB | discarded: no gain, larger, slower (1.06 ms) |
| Q6a/Q6b | Q3a with 100 / 1000 calibration images | 98.83 % / 98.90 % | 94.72 % / 94.96 % | 2.49 MB | stable: converged by 300 images |
| Q7 | fp16 | 99.17 % | 95.88 % | 4.29 MB | free accuracy but 6 % slower than fp32 on CPU (no fp16 kernels) |

Reasoning: MobileNetV2's depthwise convolutions have per-channel ranges that differ by orders of magnitude, so per-tensor scales crush small channels (Q0 vs Q1). The remaining loss was in the activation ranges, not the sensitive layers: percentile clipping helped, mixed precision did not. Dynamic quantization was never used (AI Hub rejects it; slower on CPU). QAT and AdaRound were considered and skipped (no CPU budget; AIMET has no macOS ARM build).

## 4. Pruning: what we learned (structured, torch-pruning 1.6.1, seed 0)

Width pruning removes channels (params, MACs, size all fall); depth pruning removes whole residual blocks (strongest latency lever). Each candidate: prune → adaptive BN (100 batches of the 20 % fine-tune subset) → provisional accuracy on a disjoint 10 % quick-eval subset (reference after adaptive BN: 99.37 %).

**Block scan (one block removed at a time):** accuracy drop by block — 6: 0.41, 9: 0.41, 16: 0.44, 8: 0.70, 5: 0.85, 10: 0.89, 12: 0.89, 15: 0.93, 13: 0.96, 3: 2.04 pts. The cliff is not in the same place for every layer: the early block 3 and the last-stage blocks matter most; blocks 6, 9 and 16 are nearly redundant on EuroSAT. Stacking 6+9+16 costs 1.8 pts for −17 % params and −12 % MACs; adding a fourth block (8) breaks additivity (−5.3 pts) because 8 and 9 sit in the same stage.

**Width sweep (no fine-tune):** global magnitude 10/20/30/40 % → 77.2 / 57.2 / 39.4 / 18.9 % at 1.81 / 1.44 / 1.09 / 0.83 M params; uniform control 76.3 / 65.0 / 39.2 / 21.4 %. Global and uniform converge past 20 %; uniform is slightly better at 20 %. Criterion check at global 30 %: group magnitude 39.4 % ≫ Taylor 22.7 % > BN-scale 15.8 %, i.e. on this remote-sensing data the magnitude criterion was *not* weaker (RemoteTrimmer hypothesis not supported here). Channel pruning without fine-tuning collapses on an already-compact MobileNetV2 even at 10 %; depth pruning degrades gracefully.

**Recovery fine-tune (B3, blocks 6+9+16, 1 epoch, KD from the ResNet-50 teacher, SGD lr 0.005 cosine):** see Section 5. One epoch costs ~10 minutes on this CPU, which capped the fine-tuning budget.

Considered but skipped: iterative pruning, AMC/MetaPruning search, long fine-tuning (CPU-only, one-day budget).

## 5. Pruned + int8 candidate

B3 = blocks 6, 9, 16 removed → one epoch of KD fine-tuning (5,400 images, ~14 min on CPU) → static int8 with the winning settings, re-calibrated on the pruned model. Official, all 27,000 images:

| | Baseline fp32 | Quant-only int8 (submitted) | **Pruned + int8 (B3)** |
|---|---|---|---|
| Accuracy | 99.16 % | 98.92 % | 96.52 % (fp32 before quantization: 96.49 %) |
| Worst class | PermanentCrop 95.84 % | PermanentCrop 95.24 % | PermanentCrop 89.28 % |
| Params / MACs | 2.220 M / 0.299 G | 2.220 M / 0.299 G | **1.834 M / 0.263 G** (−17 % / −12 %) |
| Size | 8.51 MB | 2.49 MB | **2.05 MB** (4.15x smaller) |
| Latency (provisional) | 2.57 ms | 0.84 ms | **0.74 ms** |
| Energy (provisional) | 0.058 J | 0.019 J | **0.016 J** |

B3 does not meet the accuracy target (≥ 98.9 %, PermanentCrop ≥ 95 %), so by the decision rule the quantization-only model is submitted and pruning is reported as a documented tradeoff point: −17 % params, −12 % MACs, 18 % smaller and ~12 % faster than the submitted model for a 2.4-point accuracy cost after one epoch. Two observations worth the judges' attention: (1) quantization costs the pruned model nothing (96.49 → 96.52 %), so the loss is entirely the pruning plus the short recovery; (2) the quick-eval accuracy after fine-tuning (96.0 %) was *below* the un-fine-tuned adaptive-BN value (97.6 %), i.e. one epoch at lr 0.005 moved the model away from its distilled optimum before converging. With the CPU-only budget we could not run the 2–3 epochs at a lower learning rate that the curve suggests would recover it; that, plus the width-pruning finalists (B1/B2) and the KD-vs-hard-label comparison, are the first things to run with more compute.

## 6. Reproducibility

`README.md` lists every command. Pipeline from the provided checkpoint: `export_onnx.py` → `measure.py` (baseline) → `quantize.py --per-channel --calib-method percentile --percentile 99.99 --types s8s8` → `check_quant.py` → `measure.py` → `timing.py` → `aihub_submit.py`. Pruning: `prune.py --config configs/<name>.json` rebuilds any candidate exactly; `stage_a.py`, `finetune.py`, `stage_c.sh`. All results, configs and AI Hub job ids are in `results/`, `configs/` and `models/*.json`; the dashboard (`results/dashboard.html`) holds every run with the tradeoff curves.
