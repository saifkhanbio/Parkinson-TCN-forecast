"""Render the locked study design as local vector and raster figures."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "study_design/figures"
W, H = 15, 21.22
INK, MUTED, LINE = "#183047", "#566777", "#7C8D9A"
PALE_BLUE, BLUE = "#EFF5FB", "#376B9A"
PALE_TEAL, TEAL = "#EDF7F4", "#287A68"
PALE_GOLD, GOLD = "#FCF5E9", "#97732A"
PALE_GRAY = "#F5F7F9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(show_status):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "saudi-pd-design-v1", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(11.69, 16.54), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    nodes, text_checks = {}, []

    def text(x, y, value, size=10, color=INK, weight="normal", **kwargs):
        return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                       va="top", ha="left", linespacing=1.35, **kwargs)

    def wrapped(value, width, size, weight="normal"):
        available = width / W * fig.get_figwidth() * fig.dpi
        font = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        result = []
        for paragraph in value.splitlines():
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                measured = renderer.get_text_width_height_descent(candidate, font, False)[0]
                if line and measured > available:
                    result.append(line)
                    line = word
                else:
                    line = candidate
            result.append(line)
        return "\n".join(result)

    def box(key, x, y, w, h, heading, body, kicker=None,
            fill=PALE_BLUE, stroke=BLUE, title_size=10.6, body_size=9.4):
        nodes[key] = (x, y, w, h)
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.085",
                               facecolor=fill, edgecolor=stroke, linewidth=1.0, zorder=3)
        ax.add_patch(patch)
        cursor = y + 0.15
        if kicker:
            text(x + 0.18, cursor, kicker, 7.7, stroke, "bold", zorder=4)
            cursor += 0.22
        title = text(x + 0.18, cursor, wrapped(heading, w - 0.36, title_size, "bold"),
                     title_size, INK, "bold", zorder=4)
        bounds = title.get_window_extent(renderer)
        cursor += bounds.height / (fig.get_figheight() * fig.dpi) * H + 0.09
        label = text(x + 0.18, cursor, wrapped(body, w - 0.36, body_size), body_size, zorder=4)
        text_checks.extend([(key, patch, title), (key, patch, label)])

    def top(key):
        x, y, w, h = nodes[key]
        return x + w / 2, y

    def bottom(key):
        x, y, w, h = nodes[key]
        return x + w / 2, y + h

    def path(points, arrow=True, color=LINE, dashed=False):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=1.05,
                linestyle="--" if dashed else "-", zorder=1)
        if arrow:
            ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                        mutation_scale=10, color=color, linewidth=1.0, zorder=2))

    def join(sources, destinations, bus_y):
        centers = [bottom(k)[0] for k in sources] + [top(k)[0] for k in destinations]
        for key in sources:
            x, y = bottom(key)
            path([(x, y), (x, bus_y)], arrow=False)
        path([(min(centers), bus_y), (max(centers), bus_y)], arrow=False)
        for key in destinations:
            x, y = top(key)
            path([(x, bus_y), (x, y)])

    text(0.55, 0.38, "Reliable Forecasting of Parkinson’s Burden\nin Saudi Arabia", 19, INK, "bold")
    text(0.55, 1.30, "Age–Sex Transfer Learning, Demographic Sensitivity, and Gulf Benchmarking", 11.1, MUTED)
    text(0.55, 1.64, "OVERALL STUDY DESIGN  /  LOCKED PROTOCOL v1.0  /  SAUDI ARABIA IS THE PRIMARY TARGET", 8.1, BLUE, "bold")

    box("regional", 0.55, 2.03, 9.55, 1.40, "Regional GBD 2023 outcome panel",
        "Saudi Arabia, Bahrain, Kuwait, Oman, Qatar, UAE and Jordan\n"
        "1990–2023 • Male and female • 11 age bands: 45–49 through 95+\n"
        "Prevalence is primary; incidence is the key secondary outcome", "CORE DATA", fill=PALE_TEAL, stroke=TEAL)
    box("support", 10.45, 2.03, 4, 1.40, "Separate supporting data",
        "UN WPP 2024; GASTAT / GCC-Stat\n"
        "Global age-standardized rates\nAvailable older GBD release points",
        "SUPPORTING INPUTS", fill=PALE_GOLD, stroke=GOLD, body_size=9.0)
    box("prepare", 0.55, 3.80, 9.55, 0.98, "Harmonize, validate and lock the analysis",
        "Check country/age/sex keys, units, source bounds and count–rate consistency\n"
        "Preserve raw files and release identifiers; record source and design hashes")
    box("windows", 0.55, 5.15, 9.55, 1.07, "Construct time-eligible training examples",
        "Learned models: eight input years → five direct horizons; at fit origin t, require u + 5 ≤ t\n"
        "Fit preprocessing only on eligible training data; keep future outcomes out of fitting")

    width = (9.55 - 0.40) / 3
    box("local", 0.55, 6.82, width, 2.10, "Saudi-only statistical models",
        "Persistence\nLog-linear trend\nDamped ETS and ARIMA\nAge-smoothed trend\nFit separately by sex",
        "COMPARATOR FAMILY 1", body_size=9.3)
    box("nonneural", 0.55 + width + 0.20, 6.82, width, 2.10, "Non-neural controls",
        "Pooled ridge and boosting\nDonor-only ridge and boosting:\nunadapted / two-coefficient\nSaudi adaptation\nMatched inputs and weights",
        "COMPARATOR FAMILY 2", body_size=9.3)
    box("tcn", 0.55 + 2 * (width + 0.20), 6.82, width, 2.10, "Regional transfer model",
        "Pretrain compact causal TCN\non six non-Saudi countries\nFreeze encoder and head\nTwo coefficients per Saudi sex\nFive-seed forecast ensemble",
        "PRIMARY MODEL", fill=PALE_TEAL, stroke=TEAL, body_size=9.3)
    box("selection", 0.55, 9.34, 9.55, 1.12, "Tune chronologically; select comparator families",
        "Inner origins: 2003 through t−5 • Family-selection origins: 2009–2013, only when completed\n"
        "Select local/non-neural comparators by sex; tune TCN settings with equal-sex loss")
    box("intervals", 0.55, 10.88, 9.55, 1.15, "Build historical forecast-error blocks and prediction intervals",
        "Reconstruct earlier fitting/tuning at each origin; save forecasts before scoring\n"
        "Joint age–sex–horizon residuals: 50% / 80% intervals; 95% is a sparse-tail sensitivity")
    half = (9.55 - 0.30) / 2
    box("primary", 0.55, 12.53, half, 1.57, "Saudi primary evaluation",
        "Fit at 2018 → forecast 2019–2023\n"
        "Primary endpoint: 2023 (horizon five)\n"
        "Equal-age mean absolute log error, by sex\n"
        "Adapted TCN vs both selected comparators",
        "PRIMARY ENDPOINT", fill=PALE_TEAL, stroke=TEAL, body_size=9.2)
    box("reliability", 0.55 + half + 0.30, 12.53, half, 1.57, "Nested reliability evaluation",
        "Repeat at origins 2014–2018\nUse only selection errors completed by t\n"
        "Error, coverage, width and interval score\nThe 2018 forecast is counted once",
        "RELIABILITY", body_size=9.2)
    box("outcomes", 0.55, 14.73, half, 1.83, "Incidence and age–sex burden",
        "Repeat forecasting for incidence\nSex-rate gaps; 65+ / 80+ burden shares\n"
        "Native-count aggregation consistency\nSupporting: mortality, YLDs, YLLs, DALYs",
        "SECONDARY / SUPPORTING", body_size=9.1)
    box("gcc", 0.55 + half + 0.30, 14.73, half, 1.83, "GCC replication and donor harm",
        "Identical evaluation in all six GCC countries\nRegional / GCC / similar / random donors\n"
        "Matched adaptation-harm comparisons\nSaudi history: 15, 20 and 29 years",
        "SECONDARY / SENSITIVITY", body_size=9.1)
    box("demography", 10.45, 14.73, 4, 1.83, "Demography and benchmarks",
        "Conditional counts; age/size/rate accounting\nUN medium vs GBD-aligned scenarios\n"
        "Separate global standardized-rate study\nAvailable release-point revisions",
        "SECONDARY / SUPPORTING", fill=PALE_GOLD, stroke=GOLD, body_size=8.8)
    box("outputs", 0.55, 17.10, 13.90, 1.22, "Report results and produce 2024–2028 projections",
        "Saudi sex-specific comparisons • GCC / donor-harm matrices • error and interval summaries • age-composition and count scenarios\n"
        "After evaluation, refit frozen procedures at origin 2023 → project 2024–2028; publish assumptions, failures and uncertainty limits",
        "FINAL DELIVERABLES", fill=PALE_TEAL, stroke=TEAL, body_size=9.2)

    path([bottom("regional"), top("prepare")])
    path([bottom("prepare"), top("windows")])
    join(["windows"], ["local", "nonneural", "tcn"], 6.50)
    join(["local", "nonneural", "tcn"], ["selection"], 9.13)
    path([bottom("selection"), top("intervals")])
    join(["intervals"], ["primary", "reliability"], 12.28)
    join(["primary", "reliability"], ["outcomes", "gcc", "demography"], 14.43)
    join(["outcomes", "gcc", "demography"], ["outputs"], 16.83)
    path([(14.45, 2.73), (14.75, 2.73), (14.75, 15.60), (14.45, 15.60)],
         color=GOLD, dashed=True)

    # Side notes are annotations, not unconnected process steps.
    text(10.57, 3.89, "EVALUATION CALENDAR", 8.5, BLUE, "bold")
    calendar = [
        ("2003 through t−5", "Completed inner validation origins"),
        ("2009–2013", "Historical family-selection origins"),
        ("2014–2018", "Nested reliability origins"),
        ("2018 → 2019–2023", "Final forecast block; endpoint 2023"),
        ("2023 → 2024–2028", "Projection fit and forecast years"),
    ]
    for i, (label, detail) in enumerate(calendar):
        y = 4.30 + i * 0.89
        ax.plot([10.63, 10.63], [y + 0.05, y + 0.53], color=BLUE, linewidth=2)
        text(10.84, y, label, 10.5, INK, "bold")
        text(10.84, y + 0.29, wrapped(detail, 3.4, 8.8), 8.8, MUTED)
    text(10.57, 9.27, "PRESPECIFIED SUCCESS RULE", 8.5, TEAL, "bold")
    text(10.57, 9.67, wrapped(
        "Lower horizon-five error for each sex against both the local and non-neural comparators. "
        "Mixed or negative results remain study findings.", 3.72, 10), 10)
    text(10.57, 11.55, "INTERPRETATION", 8.5, BLUE, "bold")
    text(10.57, 11.95, wrapped(
        "GBD outcomes are modeled estimates. Evaluation is retrospective, using revised histories.\n"
        "Eleven overlapping residual blocks are available at origin 2018; source bounds are not forecast intervals.",
        3.72, 9.4), 9.4, MUTED)

    if show_status:
        text(0.55, 18.78, "IMPLEMENTATION SNAPSHOT  /  29 SEPTEMBER 2026", 8.4, MUTED, "bold")
        stages = [
            ("0  Data and design lock", "COMPLETE", TEAL, PALE_TEAL),
            ("1  Local baselines", "DEVELOPMENT COMPLETE", TEAL, PALE_TEAL),
            ("2  Non-neural controls", "DEVELOPMENT COMPLETE", TEAL, PALE_TEAL),
            ("3  TCN and adaptation", "NEXT", BLUE, PALE_BLUE),
            ("4  Selection and intervals", "TO COMPLETE", MUTED, PALE_GRAY),
            ("5  Final / reliability tests", "PLANNED", MUTED, PALE_GRAY),
            ("6  Secondary / GCC", "PLANNED", MUTED, PALE_GRAY),
            ("7  Supporting / projections", "PLANNED", MUTED, PALE_GRAY),
        ]
        for i, (name, status, color, fill) in enumerate(stages):
            x, y = 0.55 + (i % 4) * 3.55, 19.12 + (i // 4) * 0.72
            ax.add_patch(FancyBboxPatch((x, y), 3.24, 0.59,
                                       boxstyle="round,pad=0.008,rounding_size=0.04",
                                       facecolor=fill, edgecolor="none"))
            text(x + 0.12, y + 0.08, name, 8.3, INK, "bold")
            text(x + 0.12, y + 0.34, status, 7.0, color, "bold")
    else:
        text(0.55, 18.78, "MATCHED COMPARISONS AND INFORMATION BOUNDARIES", 8.4, MUTED, "bold")
        text(0.55, 19.18,
             "Learned models share eight centered log-rate lags, last log level, sex and age indicators; balance countries, sexes, ages and windows.\n"
             "Donor fitting excludes the target country in both sexes. Saudi adaptation changes only two coefficients per sex, shared across ages.\n"
             "Forecasts and settings are saved before scoring. Population scenarios and the global standardized-rate study remain separate analyses.",
             9.2, MUTED)
    text(0.55, 20.63, "TCN: temporal convolutional network • GCC: Gulf Cooperation Council • YLD/YLL/DALY: disability and mortality burden measures", 7.5, MUTED)
    text(0.55, 20.88, "Design overview; arrows indicate analytical dependencies. Survival analysis and individual-level biological inference are outside scope.", 7.5, MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for key, patch, label in text_checks:
        p, t = patch.get_window_extent(renderer), label.get_window_extent(renderer)
        assert t.x0 >= p.x0 and t.x1 <= p.x1 and t.y0 >= p.y0 and t.y1 <= p.y1, (key, label.get_text())
    name = "overall_design_flowchart" + ("_status" if show_status else "")
    fig.savefig(OUT / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": "Parkinson’s study: overall design", "CreationDate": None})
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
    return len(nodes)


def main():
    lock_path = ROOT / "study_design/locked_v1/lock_manifest.json"
    lock = json.loads(lock_path.read_text())
    for filename, expected in lock["design_sha256"].items():
        assert sha(ROOT / filename) == expected, filename
    for name in ["local_baselines_v1", "nonneural_v1"]:
        record = json.loads((ROOT / "results" / name / "run_manifest.json").read_text())
        assert record["status"] == "complete"
        assert record["final_period_scored"] is False
    OUT.mkdir(parents=True, exist_ok=True)
    counts = [render(False), render(True)]
    record = {
        "protocol_version": "1.0", "figure_status_date": "2026-09-29",
        "locked_design_hashes_verified": True, "text_bounds_checked": True,
        "process_nodes_per_figure": counts,
        "sources_sha256": {name: sha(ROOT / name) for name in [
            "study_design/locked_v1/protocol.md", "study_design/locked_v1/design.json",
            "results/local_baselines_v1/run_manifest.json", "results/nonneural_v1/run_manifest.json"]},
        "script_sha256": sha(Path(__file__)),
        "figure_sha256": {p.name: sha(p) for p in sorted(OUT.glob("overall_design_flowchart*"))
                          if p.suffix in [".png", ".pdf", ".svg"]},
    }
    (OUT / "flowchart_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
