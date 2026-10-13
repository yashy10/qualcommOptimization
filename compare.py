"""Tabulate every result in results/ and its ratios against a baseline.

    python compare.py                           # ratios against official_baseline_onnx_fp32
    python compare.py --baseline official_baseline_pytorch
"""
import argparse
import json

from measure import RESULTS_DIR

# (result key, column header, number format)
COLUMNS = [
    ("accuracy", "acc %", ".2f"),
    ("worst_class_accuracy", "worst %", ".2f"),
    ("params_million", "params M", ".3f"),
    ("size_mb", "size MB", ".2f"),
    ("macs_giga", "MACs G", ".3f"),
    ("latency_ms", "lat ms", ".2f"),
    ("peak_ram_mb", "RAM MB", ".1f"),
    ("joules_per_inference", "J/inf", ".4f"),
]
REDUCTIONS = COLUMNS[2:]


def print_table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="official_baseline_onnx_fp32",
                        help="result name the ratios are taken against")
    args = parser.parse_args()

    results = []
    for path in RESULTS_DIR.glob("*.json"):
        result = json.loads(path.read_text())
        if "accuracy" not in result:
            continue
        if result.get("milestone_timing"):
            result.update({k: result["milestone_timing"][k] for k in ("latency_ms", "latency_median_ms", "peak_ram_mb", "joules_per_inference")})
            result["format"] += " *"
        results.append(result)
    results.sort(key=lambda result: result["measured_at"])
    if not results:
        raise SystemExit(f"no results in {RESULTS_DIR}; run measure.py first")

    print_table(
        ["name", "format", *(header for _, header, _ in COLUMNS)],
        [[r["name"], r["format"], *(format(r[key], fmt) for key, _, fmt in COLUMNS)] for r in results],
    )
    if any(r.get("milestone_timing") for r in results):
        print("\n* timing from a clean milestone session (median over rounds); others are provisional.")
    print("\nAccuracy split: " + ", ".join(sorted({r["split"] for r in results})) + ".")
    if any(r["energy_source"] == "estimate" for r in results):
        print("Energy is a codecarbon estimate, not a power-sensor reading.")

    baseline = next((r for r in results if r["name"] == args.baseline), None)
    if baseline is None:
        print(f"\nno result named {args.baseline!r}, so no ratios")
        return

    # Reductions are baseline / value, so higher is better
    print(f"\nRatios against {args.baseline} (x = baseline / value, higher is better)")
    print_table(
        ["name", "acc kept %", *(f"{header} x" for _, header, _ in REDUCTIONS)],
        [
            [
                r["name"],
                format(r["accuracy"] / baseline["accuracy"] * 100, ".2f"),
                *(format(baseline[key] / r[key], ".2f") for key, _, _ in REDUCTIONS),
            ]
            for r in results
        ],
    )


if __name__ == "__main__":
    main()
