"""Build results/dashboard.html: one self-contained page comparing every saved result.

    python dashboard.py

Reads the full results in results/ (results/smoke/ and results/archive/ are left out)
and compares each one with the ONNX fp32 baseline. measure.py runs this after every
measurement, so the page is always current. It opens offline: styles are inline and
the charts are embedded images.
"""
import base64
import io
import json
from datetime import datetime
from html import escape
from typing import NamedTuple

from measure import RESULTS_DIR, SPLIT_LABEL

BASELINE_NAME = "official_baseline_onnx_fp32"
SKIPPED_FOLDERS = {"smoke", "archive", "stageA", "timing", "aihub_profiles", "organizer_reference"}
STAGE_A_DIR = RESULTS_DIR / "stageA"
CLASS_DROP_POINTS = 1.0  # per-class drops larger than this are highlighted
BUSY_CPU_PERCENT = 10  # background load at which timing and energy deserve a second look
JITTER_FRACTION = 0.10  # latency std / mean above which the timing run was unsteady


class Metric(NamedTuple):
    key: str
    header: str
    description: str
    fmt: str
    higher_is_better: bool
    words: tuple  # (word when better, word when worse) for lower-is-better metrics
    noise_percent: float  # changes smaller than this are run-to-run noise, not a result


# Noise bands come from repeating the baseline on this laptop: the latency median and
# peak RAM moved by under 1%, the energy estimate by up to 15% when other apps were busy.
METRICS = [
    Metric("accuracy", "Accuracy (%)", "Share of all 27,000 images labelled correctly", ".2f", True, (), 0),
    Metric("worst_class_accuracy", "Worst class (%)", "Accuracy on the hardest class; rubric checks consistency",
           ".2f", True, (), 0),
    Metric("params_million", "Params (M)", "Number of learned weights; only structured pruning lowers it",
           ".3f", False, ("fewer", "more"), 0),
    Metric("size_mb", "Size (MB)", "File size on disk; quantization cuts it about 4x",
           ".2f", False, ("smaller", "larger"), 0),
    Metric("macs_giga", "MACs (G)", "Arithmetic per image; only structured pruning lowers it",
           ".3f", False, ("fewer", "more"), 0),
    Metric("latency_median_ms", "Latency (ms)", "Time per image in ONNX Runtime, median of runs",
           ".2f", False, ("faster", "slower"), 2),
    Metric("peak_ram_mb", "Peak RAM (MB)", "Most memory the process used while running",
           ".1f", False, ("less", "more"), 2),
    Metric("joules_per_inference", "Joules / inference", "Energy per image (CodeCarbon)",
           ".4f", False, ("less", "more"), 10),
]

# Chart colours: the reference data-viz palette, selected separately for each theme
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": "#3987e5"},
}

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --grid: #e1e0d9;
  --border: rgba(11, 11, 11, 0.10); --good: #0ca30c; --bad: #d03b3b; --warn: #fab219;
  --good-wash: rgba(12, 163, 12, 0.12); --bad-wash: rgba(208, 59, 59, 0.12);
  --warn-wash: rgba(250, 178, 25, 0.22); --neutral-wash: rgba(11, 11, 11, 0.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --grid: #2c2c2a;
    --border: rgba(255, 255, 255, 0.10);
    --good-wash: rgba(12, 163, 12, 0.20); --bad-wash: rgba(208, 59, 59, 0.24);
    --warn-wash: rgba(250, 178, 25, 0.18); --neutral-wash: rgba(255, 255, 255, 0.07);
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 28px 24px 48px; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 36px 0 4px; }
h3 { font-size: 14px; margin: 0 0 2px; }
p { margin: 0; color: var(--ink2); max-width: 90ch; }
code { font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; }
.scroll { overflow-x: auto; margin-top: 12px; background: var(--surface);
          border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 10px 10px; text-align: right; vertical-align: top; }
th { font-weight: 600; border-bottom: 1px solid var(--grid); min-width: 104px; }
th .desc { display: block; margin-top: 3px; font-weight: 400; font-size: 11.5px; line-height: 1.35;
           color: var(--ink2); }
th.text, td.text { text-align: left; }
td.text { min-width: 180px; }
/* The experiment name stays in view while the table scrolls sideways */
.pinned { position: sticky; left: 0; z-index: 1; background: var(--surface); }
tbody + tbody tr:first-child td, tbody tr + tr:not(.notes) td { border-top: 1px solid var(--grid); }
.value { font-variant-numeric: tabular-nums; font-size: 15px; font-weight: 600; }
.sub { display: block; font-size: 11.5px; color: var(--ink2); }
.chip { display: inline-block; margin-top: 4px; padding: 2px 7px; border-radius: 4px; font-size: 11.5px;
        font-variant-numeric: tabular-nums; background: var(--neutral-wash); }
.chip .part { white-space: nowrap; }
.chip .mark { font-weight: 700; margin-right: 3px; }
.chip.better { background: var(--good-wash); } .chip.better .mark { color: var(--good); }
.chip.worse { background: var(--bad-wash); } .chip.worse .mark { color: var(--bad); }
.name { font-weight: 600; }
.badge { display: inline-block; margin-left: 6px; padding: 1px 6px; border-radius: 4px; font-size: 11px;
         font-weight: 600; background: var(--neutral-wash); border: 1px solid var(--border); }
tr.notes td { padding-top: 0; text-align: left; }
.tag { display: inline-block; margin: 0 6px 4px 0; padding: 2px 8px; border-radius: 4px; font-size: 11.5px;
       color: var(--ink2); background: var(--neutral-wash); }
.tag.warn { color: var(--ink); background: var(--warn-wash); }
.tag.warn::before { content: "\\26A0\\00a0"; }
td.drop { background: var(--bad-wash); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; margin-top: 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card img { display: block; width: 100%; height: auto; margin-top: 8px; }
.legend { margin-top: 10px; font-size: 12.5px; }
.key { columns: 2; margin: 10px 0 0; padding: 0; list-style: none; font-size: 11.5px; color: var(--ink2); }
.key li { break-inside: avoid; }
.legend .chip { margin-right: 10px; }
footer { margin-top: 32px; font-size: 12px; color: var(--ink2); }
"""


def load_results():
    """Every full result, baseline first and the rest in the order they were run."""
    results = []
    for path in RESULTS_DIR.rglob("*.json"):
        if SKIPPED_FOLDERS & set(path.relative_to(RESULTS_DIR).parts[:-1]):
            continue
        data = json.loads(path.read_text())
        # Other JSON kept under results/, such as the organizers' reference figures, is not a result
        if "accuracy" in data:
            milestone = data.get("milestone_timing")
            if milestone:
                # Clean-session figures replace the provisional ones taken right after measuring accuracy
                for key in ("latency_ms", "latency_median_ms", "latency_p95_ms", "peak_ram_mb", "joules_per_inference"):
                    data[key] = milestone[key]
                data["latency_std_ms"] = 0.0
                data["timing_status"] = "milestone"
            results.append(data)
    results.sort(key=lambda result: (result["name"] != BASELINE_NAME, result["measured_at"]))
    return results


def change_against_baseline(metric, value, baseline_value):
    """Wording and verdict ("better", "worse" or "same") for one value against the baseline's."""
    if value == baseline_value:
        return "no change", "same"
    if metric.higher_is_better:
        points = value - baseline_value
        text = f"{points:+.2f} pts / {value / baseline_value:.1%} of baseline"
        return text.replace("-", "−"), "better" if points > 0 else "worse"
    percent = (value / baseline_value - 1) * 100
    improved = value < baseline_value
    ratio = baseline_value / value if improved else value / baseline_value
    text = (f"{percent:+.{0 if abs(percent) >= 10 else 1}f}% / "
            f"{ratio:.{1 if ratio >= 2 else 2}f}x {metric.words[0 if improved else 1]}")
    if abs(percent) < metric.noise_percent:
        return text.replace("-", "−"), "same"
    return text.replace("-", "−"), "better" if improved else "worse"


def name_cell(result, baseline, detail=""):
    # Result names are long and underscore-separated: let them wrap at the underscores
    name = escape(result["name"]).replace("_", "_<wbr>")
    badge = '<span class="badge">baseline</span>' if result is baseline else ""
    return f'<td class="text pinned"><span class="name">{name}</span>{badge}{detail}</td>'


def chip(text, verdict):
    mark = {"better": "✓", "worse": "✗", "same": "≈"}[verdict]
    # Each half stays on one line; the chip may break between them in a narrow column
    parts = " / ".join(f'<span class="part">{escape(part)}</span>' for part in text.split(" / "))
    return f'<span class="chip {verdict}"><span class="mark">{mark}</span>{parts}</span>'


def metric_cell(metric, result, baseline):
    value = result[metric.key]
    extra = title = ""
    if metric.key == "worst_class_accuracy":
        extra = f'<span class="sub">{escape(result["worst_class"])}</span>'
    elif metric.key == "latency_median_ms":
        title = (f'mean {result["latency_ms"]:.2f} ms, 95th percentile {result["latency_p95_ms"]:.2f} ms, '
                 f'std {result["latency_std_ms"]:.2f} ms')
    elif metric.key == "joules_per_inference" and result["codecarbon_mode"] == "tdp_estimate":
        extra = '<span class="sub">estimate</span>'
    if baseline is None:
        change = ""
    elif result is baseline:
        change = '<span class="chip">baseline</span>'
    else:
        change = chip(*change_against_baseline(metric, value, baseline[metric.key]))
    # The change sits straight under the value in every cell, so the chips line up along a row
    return (f'<td title="{escape(title)}"><span class="value">{value:{metric.fmt}}</span>'
            f"<br>{change}{extra}</td>")


def condition_tags(result):
    """Measurement conditions for one result, as (text, needs attention) pairs."""
    conditions = result["conditions"]
    official = result["split"] == SPLIT_LABEL and not result["smoke_test"]
    jitter = result["latency_std_ms"] / result["latency_ms"]
    tags = [
        (f'accuracy: official, all {result["images_evaluated"]:,} images', False) if official
        else (f'accuracy: smoke test on {result["images_evaluated"]:,} images, not official', True),
        (result["runtime"], False),
        (f'threads: {result["thread_setting"]}', False),
        (f'background apps: {conditions["background_cpu_percent"]:.0f}% CPU in use at start',
         conditions["background_cpu_percent"] >= BUSY_CPU_PERCENT),
        ("on battery", True) if conditions["on_ac_power"] is False else ("mains power", False),
        (f'CodeCarbon mode: {result["codecarbon_mode"]}'
         + (" (energy is estimated, not sensor-measured)" if result["codecarbon_mode"] == "tdp_estimate" else ""),
         False),
    ]
    if result.get("timing_status") == "provisional":
        tags.append(("timing: provisional (not a clean milestone run)", True))
    elif result.get("milestone_timing"):
        tags.append((f'timing: milestone session, median of {result["milestone_timing"]["rounds"]} rounds', False))
    if jitter > JITTER_FRACTION:
        tags.append((f'unsteady timing: std is {jitter:.0%} of the mean', True))
    return tags


def main_table(results, baseline):
    joules_note = ("; estimated here (tdp_estimate), not sensor-measured"
                   if any(result["codecarbon_mode"] == "tdp_estimate" for result in results) else "")
    headers = "".join(
        f'<th>{escape(metric.header)}<span class="desc">'
        f'{escape(metric.description + (joules_note if metric.key == "joules_per_inference" else ""))}</span></th>'
        for metric in METRICS
    )
    bodies = []
    for result in results:
        what = result.get("what_we_did") or "not recorded"
        measured = datetime.fromisoformat(result["measured_at"])
        cells = "".join(metric_cell(metric, result, baseline) for metric in METRICS)
        tags = "".join(
            f'<span class="tag{" warn" if attention else ""}">{escape(text)}</span>'
            for text, attention in condition_tags(result)
        )
        bodies.append(
            "<tbody><tr>"
            + name_cell(result, baseline, f'<span class="sub">{escape(result["format"])}</span>')
            + f'<td class="text">{escape(what)}</td>{cells}'
            f'<td title="{escape(result["measured_at"])}">{measured:%b %d}<span class="sub">{measured:%H:%M}</span></td>'
            f'</tr><tr class="notes"><td class="text" colspan="{len(METRICS) + 3}">{tags}</td></tr></tbody>'
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        '<th class="text pinned">Experiment<span class="desc">Result name and model format</span></th>'
        '<th class="text">What we did<span class="desc">Technique and key settings</span></th>'
        f'{headers}<th>Measured<span class="desc">When the run started</span></th>'
        f'</tr></thead>{"".join(bodies)}</table></div>'
    )


def per_class_table(results, baseline):
    classes = list((baseline or results[0])["per_class_accuracy"])
    rows = []
    for result in results:
        cells = []
        for name in classes:
            value = result["per_class_accuracy"][name]
            change = value - baseline["per_class_accuracy"][name] if baseline else 0
            dropped = change < -CLASS_DROP_POINTS
            note = ""
            if abs(change) >= 0.005:
                text = f"{change:+.2f} pts".replace("-", "−")
                note = f"<br>{chip(text, 'worse')}" if dropped else f'<span class="sub">{text}</span>'
            cells.append(f'<td class="{"drop" if dropped else ""}"><span class="value">{value:.2f}</span>{note}</td>')
        rows.append(f'<tr>{name_cell(result, baseline)}{"".join(cells)}</tr>')
    headers = "".join(f"<th>{escape(name)}</th>" for name in classes)
    return (f'<div class="scroll"><table><thead><tr><th class="text pinned">Experiment</th>{headers}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def label_points(ax, figure, points, colors):
    """Name every point, trying positions until the label is inside the plot and clear of the rest.

    Positions further from the point get a hairline leader, so a label is never left
    floating beside the wrong mark when several points sit close together.
    """
    from matplotlib.transforms import Bbox

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    plot_area = ax.get_window_extent(renderer)
    scale = figure.dpi / 72  # label offsets are in points, boxes in pixels
    occupied = []
    for x, y, _ in points:
        center_x, center_y = ax.transData.transform((x, y))
        occupied.append(Bbox.from_bounds(center_x - 6 * scale, center_y - 6 * scale, 12 * scale, 12 * scale))
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (0, -1)]
    positions = [(dx * distance, dy * distance * 0.8, distance > 9)
                 for distance in (9, 22, 36, 52) for dx, dy in directions]

    def draw(name, x, y, dx, dy, leader):
        return ax.annotate(
            name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=colors["ink"],
            ha="left" if dx > 0 else "right" if dx < 0 else "center", va="bottom" if dy > 0 else "top",
            arrowprops={"arrowstyle": "-", "color": colors["muted"], "linewidth": 0.5, "shrinkB": 4} if leader else None,
        )

    for index, (x, y, name) in enumerate(points):
        others = occupied[:index] + occupied[index + 1:]
        for dx, dy, leader in positions:
            label = draw(name, x, y, dx, dy, leader)
            box = label.get_window_extent(renderer)
            inside = (plot_area.x0 <= box.x0 and box.x1 <= plot_area.x1
                      and plot_area.y0 <= box.y0 and box.y1 <= plot_area.y1)
            if inside and not any(box.padded(2 * scale).overlaps(other) for other in others):
                break
            label.remove()
        else:
            # Nothing fits cleanly: keep the first position rather than drop the name
            label = draw(name, x, y, *positions[0])
            box = label.get_window_extent(renderer)
        occupied.append(box)


def padded_limits(values, fraction, minimum):
    low, high = min(values), max(values)
    padding = max((high - low) * fraction, minimum)
    return low - padding, high + padding


def tradeoff_chart(results, baseline, x_key, x_label, theme):
    """Accuracy against one cost metric, one named point per experiment, as a base64 PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = THEMES[theme]
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": colors["axis"],
        "axes.labelcolor": colors["ink2"],
        "xtick.color": colors["muted"],
        "ytick.color": colors["muted"],
        "xtick.labelcolor": colors["ink2"],
        "ytick.labelcolor": colors["ink2"],
    }
    with plt.rc_context(style):
        figure, ax = plt.subplots(figsize=(5.6, 3.9), dpi=200)
        figure.patch.set_facecolor(colors["surface"])
        ax.set_facecolor(colors["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.8)
        ax.grid(True, color=colors["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

        xs = [result[x_key] for result in results]
        ys = [result["accuracy"] for result in results]
        left, right = padded_limits(xs, 0.30, max(xs) * 0.12)
        ax.set_xlim(max(left, 0), right)  # sizes and times cannot be negative
        low, high = padded_limits(ys, 0.35, 0.6)
        ax.set_ylim(low, min(high, 100.3))

        if baseline:
            # Hairlines through the baseline: left of the vertical is cheaper, above the horizontal is more accurate
            ax.axvline(baseline[x_key], color=colors["axis"], linewidth=0.8, zorder=1)
            ax.axhline(baseline["accuracy"], color=colors["axis"], linewidth=0.8, zorder=1)
        others = [result for result in results if result is not baseline]
        if others:
            ax.scatter([r[x_key] for r in others], [r["accuracy"] for r in others], s=64, color=colors["series"],
                       edgecolors=colors["surface"], linewidths=1.5, zorder=3, label="Experiment")
        if baseline:
            ax.scatter([baseline[x_key]], [baseline["accuracy"]], s=70, marker="D", color=colors["ink"],
                       edgecolors=colors["surface"], linewidths=1.5, zorder=4, label="Baseline (ONNX fp32)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy (%)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=2, frameon=False, fontsize=7.5,
                  labelcolor=colors["ink2"], handletextpad=0.3, columnspacing=1.4, borderaxespad=0)
        figure.tight_layout()
        label_points(ax, figure, [(r[x_key], r["accuracy"], str(i + 1)) for i, r in enumerate(results)], colors)

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", facecolor=colors["surface"])
        plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def chart_card(results, baseline, x_key, x_label, title, reading):
    light, dark = (tradeoff_chart(results, baseline, x_key, x_label, theme) for theme in ("light", "dark"))
    listing = "; ".join(f'{i + 1} {r["name"]}: {r[x_key]:g} and {r["accuracy"]:.2f}%' for i, r in enumerate(results))
    key = "".join(f'<li><b>{i + 1}</b> {escape(r["name"])}</li>' for i, r in enumerate(results))
    return (
        f'<div class="card"><h3>{escape(title)}</h3><p>{escape(reading)}</p><picture>'
        f'<source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{dark}">'
        f'<img src="data:image/png;base64,{light}" alt="{escape(title)}. {escape(listing)}">'
        f'</picture><ol class="key">{key}</ol></div>'
    )


def stage_a_rows():
    if not STAGE_A_DIR.exists():
        return []
    rows = [json.loads(path.read_text()) for path in STAGE_A_DIR.glob("*.json")]
    return sorted(rows, key=lambda r: r["measured_at"])


def stage_a_chart(rows, x_key, x_label, title, theme):
    """Accuracy against params or MACs: one line per sweep group, points for single candidates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = THEMES[theme]
    palette = {"A2": "#2a78d6", "A3": "#eb6834", "A4": "#1baf7a", "A1": "#898781", "A5": "#6250d6", "A0": colors["ink"]}
    if theme == "dark":
        palette.update({"A2": "#3987e5", "A3": "#d95926", "A4": "#199e70", "A5": "#9085e9"})
    labels = {"A0": "A0 reference (adaptive BN only)", "A1": "A1 single block removed", "A2": "A2 block stack",
              "A3": "A3 global channels", "A4": "A4 uniform channels", "A5": "A5 other criteria @0.3"}
    with plt.rc_context({"font.size": 8, "axes.edgecolor": colors["axis"], "axes.labelcolor": colors["ink2"],
                         "xtick.color": colors["muted"], "ytick.color": colors["muted"],
                         "xtick.labelcolor": colors["ink2"], "ytick.labelcolor": colors["ink2"]}):
        figure, ax = plt.subplots(figsize=(5.6, 3.9), dpi=200)
        figure.patch.set_facecolor(colors["surface"]); ax.set_facecolor(colors["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=colors["grid"], linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
        for group in ("A0", "A1", "A2", "A3", "A4", "A5"):
            points = sorted(((r[x_key], r["accuracy"]) for r in rows if r.get("group") == group), reverse=True)
            if not points:
                continue
            xs, ys = zip(*points)
            if group in ("A2", "A3", "A4"):
                ax.plot(xs, ys, "-o", color=palette[group], linewidth=2, markersize=5,
                        markeredgecolor=colors["surface"], markeredgewidth=1, label=labels[group])
            else:
                ax.scatter(xs, ys, s=40 if group != "A0" else 70, marker="o" if group != "A0" else "D",
                           color=palette[group], edgecolors=colors["surface"], linewidths=1, zorder=4, label=labels[group])
        ax.set_xlabel(x_label); ax.set_ylabel("Quick-eval accuracy (%)")
        ax.legend(frameon=False, fontsize=7, labelcolor=colors["ink2"], loc="lower right")
        figure.tight_layout()
        buffer = io.BytesIO(); figure.savefig(buffer, format="png", facecolor=colors["surface"]); plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def block_scan_chart(rows, theme):
    """Accuracy drop when each residual block is removed alone (A1), as a bar per block."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = THEMES[theme]
    reference = next((r["accuracy"] for r in rows if r.get("group") == "A0"), None)
    scan = sorted((r for r in rows if r.get("group") == "A1"), key=lambda r: r["remove_blocks"][0])
    if reference is None or not scan:
        return None
    names = [f"features[{r['remove_blocks'][0]}]" for r in scan]
    drops = [round(reference - r["accuracy"], 2) for r in scan]
    with plt.rc_context({"font.size": 8, "axes.edgecolor": colors["axis"], "axes.labelcolor": colors["ink2"],
                         "xtick.color": colors["muted"], "ytick.color": colors["muted"],
                         "xtick.labelcolor": colors["ink2"], "ytick.labelcolor": colors["ink2"]}):
        figure, ax = plt.subplots(figsize=(5.6, 3.9), dpi=200)
        figure.patch.set_facecolor(colors["surface"]); ax.set_facecolor(colors["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, axis="y", color=colors["grid"], linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
        bars = ax.bar(names, drops, color=colors["series"], width=0.55)
        for bar, drop in zip(bars, drops):
            ax.annotate(f"{drop:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7, color=colors["ink2"])
        ax.set_ylabel("Accuracy drop (points) vs adaptive-BN reference"); ax.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        buffer = io.BytesIO(); figure.savefig(buffer, format="png", facecolor=colors["surface"]); plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def picture(light, dark, alt):
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{dark}">'
            f'<img src="data:image/png;base64,{light}" alt="{escape(alt)}"></picture>')


def stage_a_section(rows):
    if not rows:
        return ""
    reference = next((r for r in rows if r.get("group") == "A0"), None)
    body = []
    for r in rows:
        drop = f'{r["accuracy"] - reference["accuracy"]:+.2f}'.replace("-", "\u2212") if reference else ""
        body.append(
            f'<tr><td class="text pinned"><span class="name">{escape(r["name"]).replace("_", "_<wbr>")}</span>'
            f'<span class="sub">{escape(r.get("group", ""))}</span></td>'
            f'<td class="text">{escape(r["what_we_did"])}</td>'
            f'<td><span class="value">{r["accuracy"]:.2f}</span><span class="sub">{drop} pts</span></td>'
            f'<td><span class="value">{r["worst_class_accuracy"]:.2f}</span><span class="sub">{escape(r["worst_class"])}</span></td>'
            f'<td><span class="value">{r["params_million"]:.3f}</span></td><td><span class="value">{r["macs_giga"]:.3f}</span></td>'
            f'<td><span class="value">{r["size_mb"]:.2f}</span></td>'
            f'<td>{"OK" if r["export_ok"] else "FAIL"}<span class="sub">{escape(r["export_agreement"])}</span></td>'
            f'<td>{datetime.fromisoformat(r["measured_at"]):%H:%M}</td></tr>'
        )
    charts = []
    for x_key, x_label, title in (("params_million", "Params (M, thop)", "Quick-eval accuracy vs params"),
                                  ("macs_giga", "MACs (G, thop)", "Quick-eval accuracy vs MACs")):
        light, dark = (stage_a_chart(rows, x_key, x_label, title, th) for th in ("light", "dark"))
        charts.append(f'<div class="card"><h3>{escape(title)}</h3>{picture(light, dark, title)}</div>')
    scan = [block_scan_chart(rows, th) for th in ("light", "dark")]
    if scan[0]:
        charts.append(f'<div class="card"><h3>Block scan (A1): accuracy drop per removed block</h3>'
                      f'{picture(scan[0], scan[1], "block scan")}</div>')
    return f"""
<h2>Stage A pruning sweeps (provisional, no fine-tuning)</h2>
<p>Accuracy here is on the 2,700-image quick-eval subset after adaptive BN, not the official 27,000-image figure.
Params and MACs are thop counts on the PyTorch model (baseline 2.237 M / 0.326 G). Every row passed the export check
unless marked FAIL.</p>
<div class="scroll"><table><thead><tr><th class="text pinned">Candidate</th><th class="text">What we did</th>
<th>Accuracy (%)<span class="desc">quick-eval; change vs A0 reference</span></th><th>Worst class (%)</th>
<th>Params (M)</th><th>MACs (G)</th><th>Size (MB)<span class="desc">fp32 ONNX</span></th><th>Export</th><th>Time</th></tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<div class="charts">{"".join(charts)}</div>
"""


def build_dashboard(out=None):
    out = out or RESULTS_DIR / "dashboard.html"
    results = load_results()
    if not results:
        raise LookupError(f"no full results in {RESULTS_DIR}; run measure.py without --limit first")
    baseline = next((result for result in results if result["name"] == BASELINE_NAME), None)
    reference = (f"Every change is measured against <code>{BASELINE_NAME}</code>."
                 if baseline else f"<code>{BASELINE_NAME}</code> has not been measured yet, so no changes are shown.")
    noisy = ", ".join(f"{metric.header.split(' (')[0].lower()} {metric.noise_percent:g}%"
                      for metric in METRICS if metric.noise_percent)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compression dashboard</title>
<style>{CSS}</style>
</head>
<body>
<h1>Compression dashboard</h1>
<p>{len(results)} saved result{"s" if len(results) != 1 else ""}, in the order run. {reference}
Accuracy is scored on the full EuroSAT dataset, all 27,000 images, per judge instruction.</p>
<div class="legend">{chip("better than baseline", "better")}{chip("worse than baseline", "worse")}
{chip("no change, or smaller than run-to-run noise", "same")}</div>

<h2>Results</h2>
<p>Under each value: the change against the baseline, as a percentage and as a ratio.
Hover over a latency to see its mean and spread.</p>
{main_table(results, baseline)}

<h2>Accuracy by class</h2>
<p>Accuracy (%) on each of the 10 classes. Cells are highlighted where a class fell more than
{CLASS_DROP_POINTS:g} point below the baseline.</p>
{per_class_table(results, baseline)}

<h2>Tradeoffs</h2>
<p>One point per experiment. The lines cross at the baseline, so the top-left quarter is where
a model is both cheaper and at least as accurate.</p>
<div class="charts">
{chart_card(results, baseline, "size_mb", "File size (MB)", "Accuracy vs size",
            "Further left is a smaller file; higher is more accurate.")}
{chart_card(results, baseline, "latency_median_ms", "Latency (ms, median)", "Accuracy vs latency",
            "Further left is a faster answer; higher is more accurate.")}
</div>

{stage_a_section(stage_a_rows())}

<footer>Generated {datetime.now():%Y-%m-%d %H:%M} by dashboard.py from results/*.json
(results/smoke/ and results/archive/ are not included). Changes smaller than run-to-run noise
are shown as neutral: {noisy}.</footer>
</body>
</html>
"""
    out.write_text(page)
    return out


if __name__ == "__main__":
    try:
        print(f"wrote {build_dashboard()}")
    except LookupError as error:
        raise SystemExit(str(error))
