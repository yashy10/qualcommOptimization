# Quantization knowledge transfer

## Final selection update — Benson Edge

Selected run: `b3_kd_lr001_e3_int8`. Remove residual blocks 6, 9 and 16;
recover on 5,400 images for three epochs with KD, lr 0.001, batch 32,
alpha 0.7, temperature 4, seed 0, and MPS; then recalibrate static
per-channel S8S8 QDQ with percentile 99.99 on the fixed 300 images.
Full-dataset accuracy is 98.84%, worst class Pasture 97.70%, size 2.052 MiB.
The user selected this over Q3a for the smaller size and improved worst-class
accuracy despite the 0.08-point overall-accuracy tradeoff (and 0.06 points
below the original 98.9% planning target). Local timing is provisional.
A dedicated RB3 Gen 2 AI Hub profile reports 5.523 ms, 4.83x faster than the
recorded FP32 profile; job `jp2oe0kxg`. Earlier plans/results below remain
historical context, not the final submission selection.


## 1. The goal in one sentence

Convert the MobileNetV2 from 32-bit floats to 8-bit integers so the file is about 4x smaller and inference is cheaper, while losing no more than about 0.3 points of accuracy, and prove it with measurements.

## 2. Where we start (official fp32 ONNX baseline, M3 Pro, CPU)

| Metric | Baseline | What quantization does to it |
|---|---|---|
| Accuracy (all 27,000 images) | 99.16% | small drop expected |
| Worst class (PermanentCrop) | 95.84% | usually the first class to drop |
| Params | 2.220 M | no change (same count, fewer bits) |
| MACs | 0.299 G | no change (same arithmetic, in integers) |
| Size | 8.51 MB | about 3.5–4x smaller |
| Latency | 2.56 ms | probably faster on the M3, but measure it |
| Peak RAM | 88.4 MB | modest drop |
| Energy | 0.0571 J | follows latency (TDP estimate) |

Targets for the final quantized model: accuracy of at least 98.9%, PermanentCrop at no less than about 95%, size at or below 2.5 MB, and latency no worse than the baseline.

## 3. Core concepts

**The mapping.** Each real number is stored as an integer plus two parameters: real value ≈ scale × (integer − zero_point). The scale is the step size between integers, and the zero-point is the integer that stands for 0.0. With 8 bits you get 256 levels, so the question is always which range those 256 levels should cover.

**Symmetric vs asymmetric.** Symmetric fixes the zero-point at 0 and is standard for weights. Asymmetric shifts the range to fit lopsided data, which suits activations after ReLU6, since those are never negative.

**Weights vs activations.** Weights are fixed after training, so their ranges are known exactly. Activations change with every image, so their ranges have to be estimated. That's what calibration is for.

**Calibration.** Run representative images (our fixed 300) through the model, record each layer's activation values, and choose a range:

- MinMax uses the full observed range. It's simple, but one outlier stretches the range and wastes precision on everything else.
- Entropy (KL) chooses the range that best preserves the shape of the value distribution.
- Percentile clips the rarest extreme values (for example beyond the 99.99th percentile) to tighten the range.

**Granularity, the key issue for MobileNetV2.**

- Per-tensor uses one scale per layer. MobileNetV2's depthwise layers have wildly different ranges per channel, so one scale crushes the small channels to nearly zero. Research shows naive per-tensor INT8 can make MobileNetV2 collapse almost entirely.
- Per-channel uses one scale per output channel. It's the standard fix and our main setting.

**Static vs dynamic vs QAT.**

- Static PTQ (post-training quantization) precomputes activation ranges from calibration data. It's recommended for CNNs and it's what we use.
- Dynamic PTQ computes activation ranges on the fly. The packet forbids it: AI Hub rejects its operators and it's often slower on CPU.
- QAT (quantization-aware training) retrains with simulated rounding. It gives the best accuracy but needs epochs we can't afford on a CPU, so it goes in the report as "considered, skipped."

**Formats.**

- QDQ keeps the normal operators and wraps them in QuantizeLinear/DequantizeLinear pairs. The runtime fuses them into integer kernels. This is what the packet and AI Hub expect.
- QOperator uses dedicated quantized operators such as QLinearConv. We don't use it.

**Data types.** S8S8 (signed 8-bit activations and weights) is ORT's recommended first choice with QDQ. U8S8 is the alternative to compare against. `reduce_range` is only needed on older x86 chips, so skip it on the M3.

**Mixed precision.** Keep the most sensitive layers in FP32, usually the first conv (it sees raw pixels) and the final classifier (it makes the decision). It costs almost nothing in size.

**Cross-layer equalization (CLE).** The upgrade recommended by Qualcomm's research. It rescales consecutive layers so their per-channel ranges even out without changing the math. That requires replacing ReLU6 with ReLU, because ReLU6's fixed cap at 6 breaks the rescaling. High-bias absorption is a companion step. Its main payoff is making per-tensor quantization viable, and it can also help per-channel slightly.

**Preprocessing.** `quant_pre_process` runs shape inference and graph optimization, including folding BatchNorm into the convs. ORT recommends doing the optimization here rather than inside the quantizer, because that keeps accuracy problems debuggable.

**Why INT8 can be slower.** Quantize and dequantize steps add overhead, and an operator without an integer kernel falls back to FP32 with extra conversions around it. ARM chips with dot-product instructions, like the M3, usually come out ahead, but measure it.

## 4. The pipeline, step by step

1. Start from the fp32 ONNX (exported with opset 13, `dynamo=False`, input name `input`, shape (1, 3, 224, 224)).
2. Preprocess with `quant_pre_process` and save the result as a separate file.
3. Calibration reader: loads the 300 fixed calibration indices with the packet's exact transform (Resize 224, ToTensor, ImageNet normalization) and feeds them one at a time.
4. `quantize_static` with `quant_format=QDQ`, the chosen per-channel setting, calibration method, activation and weight types, and any `nodes_to_exclude`.
5. Sanity checks on every output model:
   - It loads in ONNX Runtime and accepts (1, 3, 224, 224).
   - It contains QuantizeLinear/DequantizeLinear nodes and no ConvInteger or DynamicQuantizeLinear nodes, which would mean dynamic quantization slipped in.
   - Its size is about a quarter of the fp32 file.
6. Measure accuracy on all 27,000 images, overall and per class. This step is unaffected by background apps.
7. Log to the dashboard with a "What we did" description.
8. Time only the finalists, in a clean session, alternating with the baseline (`compare.py`), median of runs.
9. AI Hub: compile and profile the best model on the chosen device. If compilation fails on duplicate `value_info` entries, strip them and re-save.

## 5. The run plan

| Run | Settings | Question it answers |
|---|---|---|
| Q0 | Per-tensor, MinMax, U8S8 | How bad is per-tensor on MobileNetV2? (expected negative result) |
| Q1 | Per-channel, MinMax, S8S8 | The main candidate. Send to AI Hub as soon as it passes. |
| Q2 | Q1 with Entropy | Does smarter range-setting help? |
| Q3 | Q1 with Percentile 99.99 and 99.999 | Does clipping outliers help? |
| Q4 | Best so far, first conv and classifier in FP32 | Does protecting the sensitive layers recover PermanentCrop, and what does it cost in size and latency? |
| Q5 | CLE + bias absorption (ReLU6 → ReLU), then per-tensor and per-channel | Does Qualcomm's recommended preprocessing help? Check FP32 accuracy after the ReLU swap first. |
| Q6 | Best config with 100 and 1000 calibration images | Is the result stable or lucky? |
| Q7 | FP16 | A cheap middle point on the tradeoff curve |

Decision rule for the winner: among the runs that meet the accuracy targets (at least 98.9% overall, PermanentCrop at about 95% or more), pick the one with the best latency and size. If two are within noise, pick the simpler one. Simpler is easier to reproduce, and that earns points.

## 6. How quantization fits with pruning

Quantization always comes last: prune, then fine-tune, then export, then quantize. Quantized weights can't be fine-tuned with normal training, and pruning changes the activation ranges, so calibration has to run on the final pruned model. Q1–Q7 on the unpruned model tell you which quantization settings work. You then reuse the winning settings on the pruned model.

## 7. Pitfalls checklist

- Calibration images come from the fixed index file and are never changed between runs.
- Always run preprocess before quantizing. Skipping it fails or quietly underperforms.
- Export with opset 13 and `dynamo=False`. The newer dynamo exporter ignores the opset setting without warning.
- Never use dynamic quantization.
- Measure ONNX, not PyTorch, with the same scripts and the same machine for baseline and final.
- Treat energy differences under about 5% as noise.
- Never let a background job run during timing.

## 8. What goes in the report

- The Q0 vs Q1 comparison, as evidence that per-channel quantization matters for depthwise layers. Cite the DFQ paper.
- The calibration-method comparison, with the reason you chose the winner.
- The mixed-precision tradeoff (accuracy gained vs size and latency cost).
- The CLE result, positive or negative.
- AdaRound and QAT as considered but skipped: AIMET doesn't support macOS ARM, and QAT needs too much CPU time.
- The accuracy-vs-size and accuracy-vs-latency curves, with your chosen point marked.
