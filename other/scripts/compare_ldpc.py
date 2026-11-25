#!/usr/bin/env python3
"""Build a with/without-LDPC comparison table and check it against an earlier sweep.

Usage:
    python scripts/compare_ldpc.py \
        --sweep_dir outputs/sweep_ldpc_XXXXXXXX \
        --ref_dir   outputs/sweep_20260411_101550 \
        --snr_list  -2 0 2 4 6 8 \
        --ldpc_n    2048
"""
import argparse
import csv
import os
import sys


def read_csv(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def fmt(v, width=10):
    try:
        return f"{float(v):.4f}".rjust(width)
    except (TypeError, ValueError):
        return str(v).rjust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--ref_dir",   required=True)
    parser.add_argument("--snr_list",  nargs="+", type=int, required=True)
    parser.add_argument("--ldpc_n",    type=int,  default=2048)
    args = parser.parse_args()

    metrics = ["psnr", "ssim", "lpips", "ter", "ber"]
    header  = (
        ["snr"]
        + [f"ref_{m}"     for m in metrics]
        + [f"no_ldpc_{m}" for m in metrics]
        + [f"ldpc_{m}"    for m in metrics]
    )

    rows = []
    for snr in args.snr_list:
        ref  = read_csv(f"{args.ref_dir}/win4096_snr{snr}_tau1.0/avg_metrics.csv")
        nold = read_csv(f"{args.sweep_dir}/no_ldpc_snr{snr}/avg_metrics.csv")
        ldpc = read_csv(f"{args.sweep_dir}/ldpc_n{args.ldpc_n}_snr{snr}/avg_metrics.csv")

        row = {"snr": snr}
        for m in metrics:
            row[f"ref_{m}"]     = ref[m]  if ref  else "N/A"
            row[f"no_ldpc_{m}"] = nold[m] if nold else "N/A"
            row[f"ldpc_{m}"]    = ldpc[m] if ldpc else "N/A"
        rows.append(row)

    # save CSV
    out_path = os.path.join(args.sweep_dir, "comparison.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"\nComparison table saved: {out_path}\n")

    # print to the terminal
    W = 10
    sep = "-" * (6 + 10 + (W + 2) * 3 + 4)

    print(f"{'SNR':>4}  {'metric':<10}  {'ref (ref.)':>{W}}  {'no_ldpc':>{W}}  {'ldpc':>{W}}")
    print(sep)

    for row in rows:
        first = True
        for m in metrics:
            snr_str = str(row["snr"]).rjust(4) if first else " " * 4
            first = False
            r  = fmt(row[f"ref_{m}"],     W)
            nl = fmt(row[f"no_ldpc_{m}"], W)
            ld = fmt(row[f"ldpc_{m}"],    W)
            print(f"{snr_str}  {m:<10}  {r}  {nl}  {ld}")
        print(sep)

    # consistency check: difference between ref and no_ldpc
    print("\n=== Reproducibility check (|no_ldpc - ref|) ===")
    print(f"{'SNR':>4}  {'metric':<10}  {'|diff|':>{W}}")
    print("-" * (6 + 10 + W + 4))
    max_diff = {}
    for row in rows:
        for m in metrics:
            r  = row[f"ref_{m}"]
            nl = row[f"no_ldpc_{m}"]
            try:
                diff = abs(float(nl) - float(r))
            except (TypeError, ValueError):
                diff = float("nan")
            max_diff[m] = max(max_diff.get(m, 0.0), diff if diff == diff else 0.0)
            print(f"{str(row['snr']):>4}  {m:<10}  {diff:{W}.6f}")
    print()
    print("Max absolute difference per metric:")
    for m in metrics:
        print(f"  {m:<10}: {max_diff.get(m, float('nan')):.6f}")


if __name__ == "__main__":
    main()
