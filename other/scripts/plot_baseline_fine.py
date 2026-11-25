#!/usr/bin/env python3
"""BER / TER vs SNR plot (for the fine sweep with 0.5 dB steps).

Usage:
    python scripts/plot_baseline_fine.py --sweep_dir outputs/sweep_baseline_fine_XXXXXXXX
"""
import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def read_metric(path: str, metric: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        return float(rows[0][metric])
    except (KeyError, ValueError):
        return None


def collect_results(sweep_dir: str, prefix: str) -> list[tuple[float, dict]]:
    """Collect (snr, metrics) from the directories matching prefix.

    Naming: {prefix}_snr{N}, e.g. baseline_snr-2.0, ldpc_snr3.5
    """
    results = []
    pattern = os.path.join(sweep_dir, f"{prefix}_snr*", "avg_metrics.csv")
    for path in glob.glob(pattern):
        snr = read_metric(path, "snr")
        if snr is None:
            continue
        ber = read_metric(path, "ber")
        ter = read_metric(path, "ter")
        results.append((snr, {"ber": ber, "ter": ter}))
    results.sort(key=lambda x: x[0])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    no_ldpc = collect_results(args.sweep_dir, "baseline")
    with_ldpc = collect_results(args.sweep_dir, "ldpc")

    if not no_ldpc and not with_ldpc:
        print("No results found.")
        return

    floor = 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Baseline (hard-decision, raw pixel) — LDPC on vs off\n"
                 f"sweep_dir: {os.path.basename(args.sweep_dir)}", fontsize=11)

    for ax, metric, ylabel in zip(axes, ["ber", "ter"], ["BER", "TER"]):
        for rows, label, color, ls, marker in [
            (no_ldpc,   "No LDPC",         "#e07b54", "-",  "o"),
            (with_ldpc, "LDPC (n=2048, R=1/2)", "#4c8cbf", "--", "s"),
        ]:
            if not rows:
                continue
            snrs = [r[0] for r in rows]
            vals = np.array([max(r[1][metric] or 0, floor) for r in rows])
            ax.semilogy(snrs, vals, marker + ls, color=color, label=label,
                        linewidth=2, markersize=6)

        ax.set_xlabel("SNR [dB]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} vs SNR")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.set_ylim(bottom=floor * 0.5)

    plt.tight_layout()
    out_path = args.out or os.path.join(args.sweep_dir, "ber_ter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
