"""Reproduce the README speedup charts from DLite paper Tables 2 and 3.

Source: attachments/DLite_Arxiv.pdf, page 8; setup: Section 5.1, page 7.
NVIDIA H800, speculative block size K = 7, relative to autoregressive decoding.
All values below are reported speedups, not acceptance lengths or derived means.

Install matplotlib, then run: python assets/plot_performance.py
Pass --preview-dir PATH to also render PNGs for visual inspection.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


METHODS = ("EAGLE-3", "DFlash", "DSpark", "DLite")
COLORS = ("#9AA6B6", "#80BBD2", "#506AA6", "#EF8335")

# Preserve benchmark order and the exact two-decimal values from the paper.
RESULTS = {
    "agentic": {
        "title": "Agentic workloads",
        "table": 3,
        "benchmarks": ["AgentMemBench", "AgentLongBench", "BFCL-v3", "SWE-Bench"],
        "ylim": 3.2,
        "models": {
            "Qwen3-4B": {
                "EAGLE-3": [1.66, 1.70, 1.58, 1.48],
                "DFlash": [1.22, 0.95, 1.44, 1.47],
                "DSpark": [1.72, 1.45, 2.11, 1.64],
                "DLite": [2.39, 2.07, 2.37, 2.68],
            },
            "Qwen3-8B": {
                "EAGLE-3": [1.77, 1.60, 1.49, 1.33],
                "DFlash": [1.80, 1.28, 1.78, 1.53],
                "DSpark": [1.83, 1.68, 1.82, 1.60],
                "DLite": [2.20, 2.31, 2.30, 2.74],
            },
        },
    },
    "long-context": {
        "title": "Long-context workloads / LongBench v2",
        "table": 2,
        "benchmarks": [
            "Multi-document QA",
            "Long in-context learning",
            "Code repository understanding",
            "Long dialogue history",
        ],
        "ylim": 4.5,
        "models": {
            "Qwen3-4B": {
                "EAGLE-3": [1.70, 1.76, 1.68, 2.23],
                "DFlash": [1.60, 1.71, 1.45, 2.62],
                "DSpark": [1.81, 2.01, 2.02, 3.16],
                "DLite": [2.18, 2.58, 2.66, 3.88],
            },
            "Qwen3-8B": {
                "EAGLE-3": [1.64, 1.87, 1.42, 2.01],
                "DFlash": [1.81, 1.86, 1.65, 2.58],
                "DSpark": [1.98, 2.28, 2.02, 3.02],
                "DLite": [2.32, 2.74, 2.53, 3.96],
            },
        },
    },
}


def plot_results(output_dir: Path, preview_dir: Path | None = None) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": "#243247",
        "axes.labelcolor": "#526174",
        "xtick.color": "#243247",
        "ytick.color": "#526174",
        "svg.fonttype": "none",
        "svg.hashsalt": "dlite-performance",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    })
    for name, result in RESULTS.items():
        fig, axes = plt.subplots(2, 1, figsize=(14, 8.4), facecolor="white")
        fig.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.09, hspace=0.49)
        fig.text(0.04, 0.95, result["title"], fontsize=23, weight="bold")
        fig.text(
            0.04, 0.905,
            "Decoding speedup vs. autoregressive decoding  |  NVIDIA H800  |  K = 7  |  Higher is better",
            fontsize=12, color="#526174",
        )
        fig.legend(
            handles=[Patch(facecolor=c, label=m) for m, c in zip(METHODS, COLORS)],
            loc="upper left", bbox_to_anchor=(0.035, 0.875), ncol=4,
            frameon=False, handlelength=1.4, columnspacing=2.2, fontsize=13,
        )
        for ax, (model, values) in zip(axes, result["models"].items()):
            ax.set_title(model, loc="left", fontsize=15, weight="bold", pad=13)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E7EBF0", linewidth=0.8)
            ax.spines["bottom"].set_color("#DCE2EA")
            ax.set_ylim(0, result["ylim"])
            ax.set_xlim(-0.55, 3.55)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}×"))
            ax.set_ylabel("Speedup", labelpad=12)
            ax.tick_params(axis="both", length=0, pad=8)
            ax.axhline(1, color="#758297", linestyle=(0, (4, 3)), linewidth=1.2)
            for i, (method, color) in enumerate(zip(METHODS, COLORS)):
                positions = [j + (i - 1.5) * 0.20 for j in range(4)]
                bars = ax.bar(positions, values[method], width=0.18, color=color, zorder=3)
                ax.bar_label(
                    bars, labels=[f"{v:.2f}" for v in values[method]],
                    padding=5, fontsize=11,
                    color="#AE4F12" if method == "DLite" else "#526174",
                    weight="bold" if method == "DLite" else "normal",
                )
            ax.set_xticks(range(4), result["benchmarks"], fontsize=11)
        fig.text(
            0.04, 0.025,
            f"Source: DLite paper, Table {result['table']} (p. 8).  Dashed line: autoregressive baseline (1.0×).",
            fontsize=11, color="#526174",
        )
        fig.savefig(
            output_dir / f"performance-{name}.svg",
            metadata={"Date": None, "Title": result["title"], "Description":
                      f"DLite paper Table {result['table']}: decoding speedups on Qwen3-4B and Qwen3-8B."},
        )
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview_dir / f"performance-{name}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    plot_results(Path(__file__).resolve().parent, args.preview_dir)
