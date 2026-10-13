"""Submit an ONNX model to Qualcomm AI Hub for a compile + profile job (plan.md, Section 4, step 9).

    python aihub_submit.py models/student_fp32.onnx --label baseline_fp32
    python aihub_submit.py models/q1_perchannel_minmax_s8s8.onnx --label q1

Strips graph.value_info entries that collide with graph outputs (AI Hub rejects them),
saves that copy under models/aihub/, submits without waiting, and appends the job ids
to results/ai_hub_jobs.json. Check progress with `python aihub_submit.py --status`.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import onnx

from common import ROOT
from measure import INPUT_SHAPE

JOBS_FILE = ROOT / "results" / "ai_hub_jobs.json"
AIHUB_DIR = ROOT / "models" / "aihub"
# Closest listed low-power Qualcomm IoT device to the Arduino UNO Q (see README)
DEFAULT_DEVICE = "Dragonwing RB3 Gen 2 Vision Kit"
# Keep the model as ONNX on the device, the format we submit to the judges
COMPILE_OPTIONS = "--target_runtime onnx"


def strip_colliding_value_info(source, target):
    model = onnx.load(source)
    outputs = {output.name for output in model.graph.output}
    kept = [info for info in model.graph.value_info if info.name not in outputs]
    removed = len(model.graph.value_info) - len(kept)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    return removed


def load_jobs():
    return json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else []


def submit(model_path, label, device_name):
    import qai_hub as hub

    device = hub.Device(device_name)
    model_path = model_path.resolve()
    cleaned = AIHUB_DIR / model_path.name
    removed = strip_colliding_value_info(model_path, cleaned)
    print(f"[{label}] stripped {removed} value_info entries colliding with outputs -> {cleaned.relative_to(ROOT)}")

    compile_job, profile_job = hub.submit_compile_and_profile_jobs(
        model=str(cleaned),
        device=device,
        name=label,
        input_specs={"input": INPUT_SHAPE},
        compile_options=COMPILE_OPTIONS,
    )
    record = {
        "label": label,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path.relative_to(ROOT)),
        "device": device_name,
        "compile_options": COMPILE_OPTIONS,
        "compile_job_id": compile_job.job_id,
        "compile_job_url": compile_job.url,
        "profile_job_id": profile_job.job_id if profile_job else None,
        "profile_job_url": profile_job.url if profile_job else None,
    }
    jobs = load_jobs()
    jobs.append(record)
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))
    print(f"[{label}] compile job {record['compile_job_id']}  {record['compile_job_url']}")
    print(f"[{label}] profile job {record['profile_job_id']}  {record['profile_job_url']}")
    return record


def status():
    import qai_hub as hub

    for record in load_jobs():
        for kind in ("compile", "profile"):
            job_id = record[f"{kind}_job_id"]
            if job_id:
                job = hub.get_job(job_id)
                print(f"{record['label']:16s} {kind:8s} {job_id}  {job.get_status().code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("--label", help="job name on AI Hub (default: the model file's stem)")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--status", action="store_true", help="print the status of every recorded job")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.model:
        submit(args.model, args.label or args.model.stem, args.device)
    else:
        parser.error("give a model to submit, or --status")


if __name__ == "__main__":
    main()
