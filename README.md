# Minimalist Edge AI Challenge: baseline

A baseline is a set of measurements of the original, unchanged model. Every later
optimization is judged against it, so it has to be measured carefully and repeatably.

Our baseline is the supplied student model, `student_mobilenetv2_distilled.pt`
(a MobileNetV2 that sorts satellite images into 10 land-use classes).

## Evaluation data

Per judge instruction, accuracy is measured on the **full EuroSAT dataset, all 27,000
images, with no `val_split`**. Every result file records this as
`"split": "full_dataset_judge_instruction"`.

Because nothing is held out, the evaluation images include the ones used for
fine-tuning and calibration below.

## Official baseline

| File | Model | Runtime |
|---|---|---|
| `results/official_baseline_onnx_fp32.json` | `models/student_fp32.onnx` | ONNX Runtime |
| `results/official_baseline_pytorch.json` | `student_mobilenetv2_distilled.pt` | PyTorch |

BASELINE_NUMBERS_PLACEHOLDER

`compare.py` takes its ratios against `official_baseline_onnx_fp32`, because the
submission is an ONNX file and ONNX Runtime is much faster than PyTorch for the very
same model.

These are our own measurements. The organizers' `baseline_metrics.json` was not
provided; the figures printed in the challenge PDF are kept apart in
`results/organizer_reference/`.

## Fixed image subsets

Both files are lists of image indices in torchvision's EuroSAT order, drawn class by
class with seed 42. `python make_subsets.py` recreates them byte for byte.

| File | Images | How it was drawn | Used for |
|---|---|---|---|
| `subsets/calibration_indices_300_seed42.json` | 300 | 30 from each class | Calibrating static quantization |
| `subsets/finetune_indices_20pct_seed42.json` | 5,400 | 20% of each class | Short recovery fine-tuning after pruning |

The 300 calibration images are all inside the fine-tuning subset, because both are the
first images of the same seeded shuffle.

To use one:

```python
import json
from torch.utils.data import Subset
from common import load_dataset

indices = json.load(open("subsets/calibration_indices_300_seed42.json"))["indices"]
calibration_set = Subset(load_dataset(), indices)
```

## How to rerun the baseline

Close other apps, plug the laptop in, and run one command at a time.

```bash
source edge-ai-env/bin/activate

# 1. Check the ONNX export still matches the original model (about 20 seconds)
python verify_export.py

# 2. Quick smoke test on 200 images, saved to results/smoke/
python measure.py models/student_fp32.onnx --name smoke --limit 200 --runs 20 --energy-seconds 3

# 3. Official baseline on all 27,000 images (about 3 minutes for ONNX, 17 for PyTorch)
python measure.py models/student_fp32.onnx --name official_baseline_onnx_fp32 --overwrite
python measure.py student_mobilenetv2_distilled.pt --name official_baseline_pytorch --overwrite

# 4. Show everything side by side
python compare.py
```

`--overwrite` is only needed because those two results already exist. Without it,
`measure.py` refuses to replace a saved result.

If `verify_export.py` ever fails, rebuild the export with `python export_onnx.py` and
run it again. The supplied `.pt` files are only ever read, never written.

## What each measurement means

| Measurement | Plain meaning | How it is measured |
|---|---|---|
| Accuracy | How often the model names the right class | One pass over all 27,000 images |
| Per-class and worst-class accuracy | Accuracy for each of the 10 classes, and the weakest one | Same pass, counted per class |
| Parameters | How many numbers the model has learned | `thop` for PyTorch, counted from the graph for ONNX |
| File size | Space the model takes on disk | Size of the file in MB |
| MACs | Amount of arithmetic for one image (multiply-accumulate operations) | `thop` for PyTorch, counted from the graph for ONNX |
| Latency | Time to answer for one image | Average of 100 timed runs after 10 warm-up runs |
| Peak RAM | Most memory the program used while answering | Process memory sampled every 10 ms with `psutil` |
| Energy per prediction | Electricity used for one answer, in joules | `codecarbon` over 15 seconds of back-to-back predictions |

Each result file also records the machine, the runtime and its version, the thread
setting, the background CPU load, the power source and the exact method used.

Things to keep in mind when reading the numbers:

- **Energy is an estimate, not a sensor reading.** On this Mac, `codecarbon` works out
  power from how busy the CPU is. Each result says so in `energy_source`.
- **Compare ONNX with ONNX.** See the note under "Official baseline".
- **MACs differ slightly between the two formats.** Export merges some layers
  (BatchNorm into convolutions), so the ONNX count is a little lower than the `thop` count.
- **Latency and energy depend on what else the laptop is doing.** Measure the baseline
  and every later model on the same machine under the same conditions.

## Files

| Path | Purpose |
|---|---|
| `common.py` | Loads the dataset and the models |
| `export_onnx.py` | Converts a PyTorch checkpoint to ONNX |
| `verify_export.py` | Confirms an ONNX export gives the same answers as PyTorch on real images |
| `measure.py` | Takes every measurement above and saves one JSON file per model |
| `compare.py` | Prints all saved results and their ratios against the baseline |
| `make_subsets.py` | Writes the calibration and fine-tuning index files |
| `subsets/` | The two fixed index files |
| `results/` | Our own full measurements, including the official baseline |
| `results/smoke/` | Quick test runs, not for reporting |
| `results/archive/` | Earlier runs on a local 20% split, kept for the record |
| `results/organizer_reference/` | Numbers from the organizers, kept apart from ours |
# qualcommOptimization
