"""Build PNG/PDF figures from the recorded evaluation JSON; no models required."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


INK, TEAL, ORANGE, PURPLE, GRAY = "#203449", "#087E8B", "#CB7425", "#7852A1", "#758493"
SEED_COLORS = [TEAL, ORANGE, PURPLE]


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#BDC9D2",
                         "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})


def polish(ax, percent=False):
    ax.grid(axis="y", color="#E5EBEF", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=3)
    if percent:
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_yticks(np.linspace(0, 1, 6))


def save(fig, output, name):
    for extension in ["png", "pdf"]:
        metadata = {"Title": name.replace("_", " ")}
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        fig.savefig(output / f"{name}.{extension}", dpi=220, metadata=metadata)
    plt.close(fig)


def overview(report, output):
    population = report["partitions"]["test"]
    rows = population["sweep_summary"]
    ks = [r["k"] for r in rows]
    seeds = report["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=.075, right=.98, bottom=.29, top=.80, wspace=.26)
    fig.suptitle("Motif localization and teacher fidelity", x=.075, y=.97, ha="left", fontsize=19, fontweight="bold")
    fig.text(.075, .905, f"Final evaluation on unseen synthetic ancestor families · {population['n_sequences']:,} test sequences · "
             f"{population['n_families']} families", fontsize=10.5, color=GRAY)
    fig.text(.075, .865, f"{len(seeds)} explainer seeds · one fixed teacher · Lines: seed means; faint points: individual runs",
             fontsize=9.5, color=GRAY)
    specifications = [
        ("localization", "positive_any_motif_overlap_rate", "Any motif overlap", TEAL),
        ("localization", "positive_full_motif_window_recovery_rate", "Complete motif window", ORANGE),
        ("keep", "teacher_agreement", "Class agreement", TEAL),
        ("keep", "balanced_teacher_agreement", "Balanced class agreement", PURPLE),
    ]
    for group, metric, label, color in specifications:
        ax = axes[0 if group == "localization" else 1]
        for j, seed in enumerate(seeds):
            runs = [r for r in population["sweep_runs"] if r["seed"] == seed]
            values = [r["selection" if group == "localization" else "keep"][metric] for r in runs]
            ax.scatter(np.array(ks) + (j-(len(seeds)-1)/2)*.09, values, s=14, color=color, alpha=.35, edgecolors="none")
        ax.plot(ks, [r[group][metric]["mean"] for r in rows], "o-", color=color, linewidth=2, markersize=4.5, label=label)
    axes[0].plot(ks, [r["uniform_random_any_overlap"] for r in rows], "--", color=GRAY,
                 linewidth=1.4, label="Uniform-random overlap (exact)")
    axes[1].axhline(population["baselines"]["constant_class_teacher_agreement"], color=GRAY, linestyle="--",
                    linewidth=1.3, label="Constant-class agreement")
    axes[1].axhline(.5, color=GRAY, linestyle=":", linewidth=1.3, label="Constant-class balanced agreement")
    for ax, title, ylabel in zip(axes, ["A   Localization on motif-positive sequences", "B   Imitation on all test sequences"],
                                ["Positive sequences recovered", "Agreement with teacher"]):
        polish(ax, percent=True)
        ax.set(title=title, xlabel="Selected positions (k)", ylabel=ylabel, xticks=ks, xlim=(.6, max(ks)+.4))
        ax.title.set_fontsize(10.5)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.23), frameon=False, fontsize=8.7, handlelength=2.2)
    save(fig, output, "test_overview")


def comparison(report, output):
    metrics = [("teacher_agreement", "Class agreement ↑"),
               ("balanced_teacher_agreement", "Balanced agreement ↑"), ("teacher_kl", "Teacher KL ↓")]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8))
    fig.subplots_adjust(left=.095, right=.97, top=.80, bottom=.17, hspace=.55, wspace=.30)
    fig.suptitle("Matching the approximator’s training masks", x=.095, y=.97, ha="left", fontsize=18, fontweight="bold")
    fig.text(.095, .913, "Frozen teacher and top-5 selectors · identical starting weights and batches · 15 additional epochs per arm",
             fontsize=9.6, color=GRAY)
    kl_max = max(r["metrics"]["teacher_kl"] for p in report["partitions"].values() for r in p["mask_comparison_runs"])
    for row, partition in enumerate(["validation", "test"]):
        population = report["partitions"][partition]
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            for j, seed in enumerate(report["seeds"]):
                values = [next(r["metrics"][metric] for r in population["mask_comparison_runs"]
                               if r["seed"] == seed and r["mode"] == mode) for mode in ["soft", "hard"]]
                ax.plot([0, 1], values, "o-", color=SEED_COLORS[j % len(SEED_COLORS)], alpha=.8,
                        linewidth=1.2, markersize=5)
            for x, mode in enumerate(["soft", "hard"]):
                stats = population["mask_comparison_summary"][mode][metric]
                ax.errorbar(x+.045, stats["mean"], yerr=stats["sample_sd"], fmt="s", color=INK,
                            markersize=5, capsize=3, elinewidth=1.4, zorder=4)
            polish(ax, percent=col < 2)
            if col == 2:
                ax.set_ylim(0, np.ceil((kl_max+.03)*10)/10)
                ax.set_ylabel("KL divergence (nats)", fontsize=9)
            else:
                ax.set_ylabel("Agreement", fontsize=9)
            ax.set_xlim(-.25, 1.3)
            ax.set_xticks([0, 1], ["Soft training", "Hard training"])
            ax.set_title(f"{partition.title()} · {title}", fontsize=10, loc="left", pad=10)
    handles = [Line2D([0], [0], color=SEED_COLORS[j % 3], marker="o", label=f"Seed {seed}", linewidth=1.2)
               for j, seed in enumerate(report["seeds"])]
    handles.append(Line2D([0], [0], marker="s", color=INK, linestyle="none", label="Mean ± sample SD"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .065), ncol=4, frameon=False, fontsize=9)
    fig.text(.095, .025, "Both arms evaluated with exact top-5 masks. Paired lines show each seed; sample SDs are not confidence intervals.",
             fontsize=8.7, color=GRAY)
    save(fig, output, "mask_comparison")


def examples(report, output):
    selected_examples = report["examples"]
    fig = plt.figure(figsize=(13.5, 8))
    grid = fig.add_gridspec(len(selected_examples)*2, 1, height_ratios=[3, .65]*len(selected_examples),
                           left=.065, right=.98, top=.83, bottom=.17, hspace=.70)
    fig.suptitle("What the selector chooses", x=.065, y=.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(.065, .915, "Original L2X · five selected bases · seed 0 · test examples fixed by sample ID before scoring",
             fontsize=10.5, color=GRAY)
    for row, example in enumerate(selected_examples):
        scores = np.asarray(example["scores"])
        chosen = np.asarray(example["selected_positions"])
        axis = fig.add_subplot(grid[row*2])
        dna = fig.add_subplot(grid[row*2+1], sharex=axis)
        for start, end in example["motif_windows"]:
            axis.axvspan(start-.5, end+.5, color=TEAL, alpha=.13, linewidth=0)
            dna.plot([start-.45, end+.45], [.04, .04], color=TEAL, linewidth=3.5, solid_capstyle="butt")
        axis.plot(np.arange(len(scores)), scores, color=GRAY, linewidth=1.2)
        axis.scatter(chosen, scores[chosen], color=ORANGE, s=27, zorder=4)
        label = "motif positive" if example["label"] else "motif negative"
        axis.set_title(f"{chr(65+row)}   {example['sample_id']} · {label}", loc="left", fontsize=9.5, pad=9)
        axis.set_title(f"Teacher P(1) {example['teacher_positive_probability']:.2f}    "
                       f"Q(top-5) P(1) {example['keep_positive_probability']:.2f}", loc="right", fontsize=9, fontweight="normal", pad=9)
        axis.set_ylabel("Score", fontsize=8.5)
        axis.yaxis.set_major_locator(MaxNLocator(3))
        axis.set_xlim(-1, len(scores))
        axis.set_xticks([0, 24, 49, 74, 99])
        axis.tick_params(axis="both", labelsize=8)
        axis.grid(axis="y", color="#E5EBEF", linewidth=.6)
        for position, base in enumerate(example["sequence"]):
            selected = position in chosen
            if selected:
                dna.add_patch(Rectangle((position-.46, .12), .92, .77, facecolor=ORANGE, alpha=.20, linewidth=0))
            dna.text(position, .50, base, ha="center", va="center", fontfamily="DejaVu Sans Mono", fontsize=7.5,
                     color=INK if not selected else "#8C4813", fontweight="bold" if selected else "normal")
        dna.set_ylim(0, 1)
        dna.axis("off")
    handles = [Line2D([0], [0], color=GRAY, label="Selector logits"),
               Line2D([0], [0], color=ORANGE, marker="o", linestyle="none", label="Exact top-5 selection"),
               Patch(facecolor=TEAL, alpha=.25, label="Known motif window")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .065), ncol=3, frameon=False, fontsize=9)
    fig.text(.065, .025, "Two positive and one negative sequence, chosen without ranking selection quality. DNA positions are zero-indexed.",
             fontsize=8.8, color=GRAY)
    save(fig, output, "selection_examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/final_evaluation.json")
    parser.add_argument("--out_dir", default="figures")
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.out_dir)
    report = json.loads(source.read_text())
    if report.get("status") != "complete":
        raise ValueError("Figures require a complete evaluation report.")
    output.mkdir(parents=True, exist_ok=True)
    setup()
    overview(report, output)
    comparison(report, output)
    examples(report, output)
    manifest = {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "protocol_fingerprint": report["protocol_fingerprint"],
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "formats": ["PNG at 220 dpi", "PDF with embedded fonts"],
                "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(output.iterdir()) if p.suffix in {".png", ".pdf"}}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved three figures in PNG and PDF to", output)


if __name__ == "__main__":
    main()
