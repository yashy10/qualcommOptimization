# PLAN_PRUNING.md — Structured Pruning Plan (MobileNetV2 / EuroSAT)

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


Read this together with PLAN.md (quantization) and CLAUDE.md (standing rules).
If anything here conflicts with CLAUDE.md, CLAUDE.md wins. Stop and report at the
end of every stage. Never start the next stage without the user's go-ahead.

---

## 1. Goal

Make the model physically smaller (fewer parameters, fewer MACs, smaller file)
and faster (lower latency and energy) with **width pruning** (remove channels)
and **depth pruning** (remove whole residual blocks), recover accuracy with a
short fine-tune, then quantize the result with the winning quantization settings.

Quantization alone does not change params or MACs. Pruning is the only lever for
those two numbers, and depth pruning is the strongest lever for latency/energy.

## 2. Baseline (official, fp32 ONNX, M3 Pro, CPU)

| Metric | Baseline |
|---|---|
| Accuracy (all 27,000 images) | 99.16% |
| Worst class | PermanentCrop 95.84% |
| Params | 2.220 M (ONNX) / 2.237 M (thop on PyTorch — use thop for baseline AND pruned) |
| MACs | 0.326 G (thop on PyTorch — same method for pruned) |
| Size | 8.51 MB |
| Latency | 2.56 ms |
| Peak RAM | 88.4 MB |
| Energy | 0.0571 J (TDP estimate) |

## 3. Targets for the final pruned + INT8 model

- Accuracy >= 98.9% on all 27,000 images (or within ~0.3 points of the
  quantization-only model).
- PermanentCrop >= ~95%.
- Params reduced by >= ~2x if accuracy allows.
- Latency and energy lower than the quantization-only model.

## 4. Research basis (cite in the report)

- **DepGraph / Torch-Pruning (CVPR 2023):** structured pruning must remove
  coupled parameters together; a simple group-level magnitude criterion performs
  well once dependencies are handled.
- **Torch-Pruning docs:** `pruning_ratio` is per CHANNEL; params drop by about
  1-(1-p)^2 (p=0.3 ≈ 51% of params). Global pruning can over-prune single layers;
  `isomorphic=True` mitigates this. `round_to=8` recommended.
- **AMC / MetaPruning:** non-uniform (per-layer) ratios beat uniform width
  shrinkage on MobileNetV2.
- **LayerPrune:** removing whole layers/blocks gives larger real latency
  reduction than filter pruning, especially in residual networks.
- **EagleEye (ECCV 2020):** adaptive BN (recomputing BN statistics after
  pruning) strongly predicts post-fine-tune accuracy, so candidates can be
  ranked without fine-tuning.
- **RemoteTrimmer (2024, EuroSAT):** channel-importance differences are less
  pronounced on remote-sensing imagery, so plain weight magnitude is a weaker
  signal; test a data- or BN-based criterion against it.
- **Recovery studies (2025-2026):** small data suffices at moderate ratios;
  combine hard labels with KD rather than KD alone.

## 5. Standing rules for pruning

1. **Library:** torch-pruning, version pinned in requirements.txt.
2. **Autograd ON** when building the dependency graph (never inside
   `torch.no_grad()`).
3. **Never prune the classifier** (`ignored_layers` = final Linear).
4. **round_to=8** for all channel pruning.
5. **Determinism:** seed 0 everywhere. `prune.py` must regenerate any pruned
   model exactly from the original checkpoint + a JSON config.
6. **Saving:** `torch.save(model)` (whole model, not state_dict) +
   `configs/<name>.json`. Load with `torch.load(..., weights_only=False)`.
7. **Adaptive BN after every prune and before every measurement:** reset BN
   running stats, run 100 batches (batch 32) from the fine-tune subset in
   train mode with no grad, then switch to eval.
8. **Export check for every dashboard candidate:** ONNX opset 13,
   `dynamo=False`, input name `input`, shape (1,3,224,224), outputs 10 scores,
   predictions agree with the PyTorch model on 500 images.
9. **Params/MACs** always from thop on the PyTorch model (same as baseline).
10. **No timing** in Stages A and B. Timing only in Stage C via `compare.py`.
11. **Never fine-tune on the quick-eval subset.** Never overwrite results.

## 6. Fixed data subsets (create once, save indices, never change)

| File | Content | Use |
|---|---|---|
| `finetune_subset_20.json` | 20% of all images, stratified by class, seed 0 | Adaptive BN, fine-tuning, teacher-logit cache, Taylor gradients |
| `quickeval_subset_10.json` | 10% stratified, seed 0, **disjoint from the fine-tune subset** | Fast provisional scoring in Stage A |
| (derived) "unseen 80%" | All images NOT in the fine-tune subset | Honesty check after fine-tuning |

Official accuracy is always on all 27,000 images (judge instruction).

## 7. Stage A — cheap sweeps, NO fine-tuning (target: done by ~1:30 pm)

All candidates: prune → adaptive BN → quick-eval (provisional) → log to dashboard.

**A1 Block scan.** For each inverted-residual block with a residual connection
(stride 1, in_channels == out_channels; 10 blocks in torchvision MobileNetV2),
replace the block with `nn.Identity()`, adaptive BN, quick-eval. Rank blocks by
accuracy drop. Report a table + bar chart.

**A2 Block stack.** Cumulatively remove the 1, 2, 3, 4, 5, 6 least-damaging
blocks from A1. Report accuracy, params, MACs per step.

**A3 Channel sweep (global).** `tp.pruner.BasePruner`,
`GroupMagnitudeImportance(p=2)`, `global_pruning=True`, `isomorphic=True`,
`round_to=8`, classifier ignored, channel `pruning_ratio` 0.1 / 0.2 / 0.3 / 0.4.
Report the ACTUAL param reduction for each.

**A4 Uniform control.** Same as A3 with `global_pruning=False`.

**A5 Criterion check** at channel ratio 0.3 (global):
GroupMagnitude vs `BNScaleImportance` vs `TaylorImportance`
(Taylor: accumulate cross-entropy gradients over 10 batches of the fine-tune
subset before pruning).

**A6 Timing probes (for planning only, not results):**
- One fine-tune epoch (batch 32, SGD) on the 20% subset for the A3 p=0.3 model.
- Caching teacher (ResNet-50) logits on the 20% subset.

**Stage A report:** tables for A1–A5 (accuracy, worst class, params, MACs,
size), one accuracy-vs-params plot with all curves, A6 times, and 2–3
recommended finalists including at least one **width + depth** combination.
STOP.

## 8. Stage B — fine-tune finalists (target: done by ~2:15 pm)

**Finalists (decided with the user after Stage A), typically:**
- B1: best channel ratio alone
- B2: best channel ratio + removable blocks (width + depth)
- B3: one more aggressive combination

**Fine-tune recipe (all finalists):**
- Data: `finetune_subset_20.json`, **no augmentation** (keeps cached teacher
  logits valid).
- Optimizer: SGD, momentum 0.9, lr 0.005, cosine decay to 0, weight decay 4e-5,
  batch 32.
- Epochs: 1–2 (choose from the A6 timing; cap ~10 min per finalist).
- Loss (after the KD comparison): `(1-α)·CE(student, labels) +
  α·T²·KL(softmax(student/T) || softmax(teacher/T))`, T=4, α=0.7.
- Teacher logits cached once to `cache/teacher_logits_ft20.pt`.

**KD comparison (on ONE finalist only, usually B2):**
hard labels only vs hard labels + KD, same epochs and lr. Use the winner for
the other finalists. Report both — either result goes in the report.

**After fine-tuning, for each finalist:**
- Adaptive BN not needed (BN stats updated by training), but run model.eval().
- Accuracy + per-class on ALL 27,000 images (official).
- Accuracy on the unseen 80% (honesty check). Flag if the gap > 0.3 points.
- Params, MACs, size (fp32 ONNX).
- Export check.

**Stage B report:** before/after fine-tune table, KD vs hard-label result,
per-class drops > 1 point, recommendation. STOP.

## 9. Stage C — quantize + full benchmark (target: done by ~2:45 pm)

1. Export each surviving finalist to fp32 ONNX (opset 13, dynamo=False).
2. Quantize with the **winning settings from PLAN.md** (static QDQ,
   per-channel, chosen calibration method, S8S8/U8S8, exclusions), using the
   fixed calibration indices. Re-calibrate on the pruned model — never reuse
   old ranges.
3. Sanity checks (`check_quant.py`): loads, QDQ nodes present, no dynamic ops,
   size ratio.
4. Accuracy + per-class on all 27,000 images.
5. **Full benchmark** in a clean session (apps closed, plugged in) with
   `compare.py`, alternating baseline vs candidate, median of runs: latency,
   peak RAM, energy.
6. Submit the chosen model to AI Hub (same device as before), record job IDs.

**Decision rule:** among INT8 candidates meeting the targets in Section 3, pick
the best combined latency + size. If two are within noise, pick the simpler
one. If no pruned candidate meets the targets, submit the quantization-only
model and report pruning as a documented negative result.

## 10. Pitfalls checklist

- Channel ratio ≠ parameter ratio (≈ 1-(1-p)^2).
- Global pruning without `isomorphic=True` can gut a single layer.
- Forgetting adaptive BN makes un-fine-tuned accuracy look far worse than it is.
- Building the dependency graph under `no_grad` breaks torch-pruning.
- `load_state_dict` into a fresh `mobilenet_v2()` will fail for pruned models.
- Removing a block that changes stride or channel count breaks the network —
  only remove blocks with a residual connection.
- Augmentation + cached teacher logits = mismatched targets. No augmentation.
- Always fine-tune BEFORE quantizing; always re-calibrate after pruning.

## 11. Logging (dashboard)

Every candidate row: name, stage, "What we did" (mode, ratio, criterion,
blocks removed, fine-tune recipe), accuracy (provisional or official), worst
class, params, MACs, size, export OK, timestamp. Plots: accuracy vs params
(all Stage A curves), accuracy vs MACs, block-scan bar chart, and the final
tradeoff curve (baseline → quant-only → pruned+INT8 finalists).

## 12. What goes in the report

- Block-scan chart: which blocks are redundant on EuroSAT (answers "is the
  cliff in the same place for every layer?").
- Width curve (global vs uniform) and depth curve; the chosen point and why.
- Criterion comparison (magnitude vs BN-scale vs Taylor) and the
  RemoteTrimmer hypothesis.
- Hard-label vs KD recovery result.
- Unseen-80% vs full-dataset accuracy (shows recovery is not memorization).
- Considered but skipped: iterative pruning, AMC/MetaPruning search,
  RemoteTrimmer's attention-based pruning, long fine-tuning — all exceed the
  CPU-only, one-day budget.
- Final tradeoff curve with the submitted model marked.
