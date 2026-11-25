#!/usr/bin/env python3
"""BER / TER vs SNR plot (with vs without LDPC).

Usage:
    python scripts/plot_ldpc_ber_ter.py \
        --sweep_dir outputs/sweep_baseline_ldpc_20260411_162708 \
        --snr_list -2 0 2 4 6 8 \
        --ldpc_n 2048
"""
import argparse
import csv
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--snr_list", nargs="+", type=int, required=True)
    parser.add_argument("--ldpc_n", type=int, default=2048)
    parser.add_argument("--out", default=None, help="output file (default: sweep_dir/ber_ter.png)")
    args = parser.parse_args()

    snrs = sorted(args.snr_list)

    no_ber, no_ter, ld_ber, ld_ter = [], [], [], []

    for snr in snrs:
        no_path = f"{args.sweep_dir}/baseline_ldpcfalse_snr{snr}/avg_metrics.csv"
        ld_path = f"{args.sweep_dir}/baseline_ldpctrue_snr{snr}/avg_metrics.csv"

        no_ber.append(read_metric(no_path, "ber"))
        no_ter.append(read_metric(no_path, "ter"))
        ld_ber.append(read_metric(ld_path, "ber"))
        ld_ter.append(read_metric(ld_path, "ter"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Baseline (hard-decision) — LDPC on vs off", fontsize=13)

    for ax, (no_vals, ld_vals, ylabel) in zip(
        axes,
        [
            (no_ber, ld_ber, "BER"),
            (no_ter, ld_ter, "TER"),
        ],
    ):
        no_np = np.array([v if v is not None else np.nan for v in no_vals])
        ld_np = np.array([v if v is not None else np.nan for v in ld_vals])

        # floor tiny values so log-scale looks clean
        floor = 1e-5
        no_plot = np.where(no_np < floor, floor, no_np)
        ld_plot = np.where(ld_np < floor, floor, ld_np)

        ax.semilogy(snrs, no_plot, "o-", color="#e07b54", label="No LDPC", linewidth=2, markersize=7)
        ax.semilogy(snrs, ld_plot, "s--", color="#4c8cbf", label=f"LDPC (n={args.ldpc_n}, R=1/2)", linewidth=2, markersize=7)

        # mark floor points with open markers
        floored_no = no_np < floor
        floored_ld = ld_np < floor
        if floored_no.any():
            ax.semilogy(
                [snrs[i] for i in range(len(snrs)) if floored_no[i]],
                [floor] * floored_no.sum(),
                "v", color="#e07b54", markersize=9, alpha=0.5,
            )
        if floored_ld.any():
            ax.semilogy(
                [snrs[i] for i in range(len(snrs)) if floored_ld[i]],
                [floor] * floored_ld.sum(),
                "v", color="#4c8cbf", markersize=9, alpha=0.5,
                label=f"LDPC {ylabel}=0 (floor)",
            )

        ax.set_xlabel("SNR [dB]", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel + " vs SNR")
        ax.set_xticks(snrs)
        ax.set_xlim(min(snrs) - 0.5, max(snrs) + 0.5)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    plt.tight_layout()

    out_path = args.out or os.path.join(args.sweep_dir, "ber_ter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
