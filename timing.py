"""Milestone timing: latency, peak RAM and energy for a few models in one clean session.

    python timing.py official_baseline_onnx_fp32 q3a_perchannel_pct9999_s8s8 q7_fp16 --rounds 5

Runs the models in alternation (round 1: A B C, round 2: A B C, ...) so slow drift in the
machine's state affects all of them equally, and takes the median over rounds. Refuses to
start if background CPU load is high. Writes results/timing/milestone_<time>.json and adds a
`milestone_timing` block to each result file (the provisional figures are kept alongside);
the dashboard and compare.py prefer the milestone figures when present.
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import psutil

from measure import RESULTS_DIR, ROOT

MAX_BACKGROUND_CPU = 10.0
TIMED_KEYS = ["latency_ms", "latency_median_ms", "latency_p95_ms", "peak_ram_mb", "joules_per_inference"]


def one_run(result, runs, energy_seconds):
    """Runtime metrics for one model in a fresh process (measure.py's hidden --runtime-out mode)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "runtime.json"
        command = [sys.executable, str(ROOT / "measure.py"), str(ROOT / result["model"]), "--runs", str(runs),
                   "--energy-seconds", str(energy_seconds), "--runtime-out", str(out)]
        if result["format"] == "pytorch":
            command += ["--arch", "mobilenet_v2"]
        child = subprocess.run(command, capture_output=True, text=True)
        if child.returncode != 0:
            sys.exit(f"timing {result['name']} failed:\n{child.stderr[-800:]}")
        return json.loads(out.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="result names in results/ (the baseline first)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100, help="timed inferences per round")
    parser.add_argument("--energy-seconds", type=float, default=15)
    args = parser.parse_args()

    results = {name: json.loads((RESULTS_DIR / f"{name}.json").read_text()) for name in args.names}
    battery = psutil.sensors_battery()
    load = psutil.cpu_percent(interval=3)
    print(f"background CPU {load:.0f}% | {'mains' if battery is None or battery.power_plugged else 'BATTERY'} power")
    if load > MAX_BACKGROUND_CPU:
        sys.exit(f"background CPU load {load:.0f}% is above {MAX_BACKGROUND_CPU:.0f}%; close other apps and retry")

    rounds = {name: [] for name in args.names}
    for round_number in range(1, args.rounds + 1):
        for name in args.names:
            metrics = one_run(results[name], args.runs, args.energy_seconds)
            rounds[name].append(metrics)
            print(f"round {round_number} {name:32s} median {metrics['latency_median_ms']:.2f} ms | "
                  f"RAM {metrics['peak_ram_mb']:.1f} MB | {metrics['joules_per_inference']:.4f} J")

    stamp = datetime.now().isoformat(timespec="seconds")
    session = {
        "measured_at": stamp,
        "rounds": args.rounds, "runs_per_round": args.runs, "energy_seconds": args.energy_seconds,
        "background_cpu_percent_at_start": load,
        "on_ac_power": battery.power_plugged if battery else None,
        "models": {},
    }
    for name in args.names:
        medians = {key: round(statistics.median(r[key] for r in rounds[name]), 5) for key in TIMED_KEYS}
        spread = {key: round(max(r[key] for r in rounds[name]) - min(r[key] for r in rounds[name]), 5)
                  for key in TIMED_KEYS}
        session["models"][name] = {"median_over_rounds": medians, "range_over_rounds": spread, "rounds": rounds[name]}
        result = results[name]
        result["milestone_timing"] = {"measured_at": stamp, "rounds": args.rounds, **medians,
                                      "range_over_rounds": spread,
                                      "energy_source": rounds[name][0]["energy_source"],
                                      "codecarbon_mode": rounds[name][0]["codecarbon_mode"]}
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))

    out_dir = RESULTS_DIR / "timing"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"milestone_{stamp.replace(':', '-')}.json"
    out.write_text(json.dumps(session, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")
    baseline = session["models"][args.names[0]]["median_over_rounds"]
    for name in args.names:
        m = session["models"][name]["median_over_rounds"]
        print(f"{name:32s} {m['latency_median_ms']:.2f} ms ({baseline['latency_median_ms'] / m['latency_median_ms']:.2f}x) | "
              f"RAM {m['peak_ram_mb']:.1f} MB | {m['joules_per_inference']:.4f} J "
              f"({baseline['joules_per_inference'] / m['joules_per_inference']:.2f}x)")

    try:
        from dashboard import build_dashboard

        build_dashboard()
    except Exception as error:
        print(f"dashboard not updated: {error!r}")


if __name__ == "__main__":
    main()
