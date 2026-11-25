"""Print the paper table "Impact of Window Size W on TER" from the window sweep.

Usage:
    python scripts/paper_window_table.py results/sweeps/sweep_20260416_133456__window

TER is rounded to 4 decimals. 14 of the 16 cells match the paper exactly; the two
cells at 1 dB for W=10 and W=128 differ by 0.0001 because the paper truncated
instead of rounding (0.348882 -> paper 0.3488 / script 0.3489,
0.263386 -> paper 0.2633 / script 0.2634).
"""

import pathlib
import sys

import pandas as pd

SNRS = [1, 3, 5, 7]
WINDOWS = [10, 128, 512, 4096]


def load(root: pathlib.Path) -> pd.DataFrame:
    files = sorted(root.rglob("avg_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No avg_metrics.csv found under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["use_llm"].astype(bool) & ~df["use_ldpc"].astype(bool)]
    # The table uses QPSK only
    if "num_bits_per_symbol" in df.columns:
        df = df[df["num_bits_per_symbol"].fillna(2) == 2]
    return df


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/sweeps/sweep_20260416_133456__window")
    df = load(root)
    table = df.pivot_table(index="snr", columns="window_size", values="ter").loc[SNRS, WINDOWS]
    print("Eb/N0 [dB] & " + " & ".join(f"$W={w}$" for w in WINDOWS) + r" \\")
    for snr, row in table.iterrows():
        cells = " & ".join(f"{v:.4f}" for v in row)
        print(f"{snr} & {cells} " + r"\\")


if __name__ == "__main__":
    main()
