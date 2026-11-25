"""
analyze_sweep.py
Aggregate and plot the results of a parameter sweep (directories of avg_metrics.csv).

Usage (from the repository root):
    python scripts/analyze_sweep.py --results_dir results/sweeps/sweep_20260413_095050_ --eb_n0
    python scripts/analyze_sweep.py --results_dir <sweep_dir> --fix_window 4096
    python scripts/analyze_sweep.py --results_dir <sweep_dir> --fix_snr 4

The exact options used for the paper figures are in scripts/reproduce_paper.sh.
"""

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns



METRICS = ["ter", "ber", "psnr", "ssim", "lpips", "clip_similarity", "fid"]
LOG_SCALE_METRICS = {"ter", "ber"}  # metrics shown on a log-scale y axis
METRIC_UNITS = {"psnr": "(dB)"}
METRIC_LABELS = {
    "clip_similarity": "CLIP Score",
    "ter": "TER",
    "ber": "BER",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "fid": "FID",
}
# metrics for which the maximum SNR gain is annotated
GAIN_ANNOTATE_METRICS = {"psnr", "lpips", "clip_similarity"}


def metric_ylabel(metric: str) -> str:
    label = METRIC_LABELS.get(metric, metric.upper())
    unit = METRIC_UNITS.get(metric)
    return f"{label} {unit}" if unit else label

# -----------------------------------------------------------------------
# SNR axis (channel.snr is already Eb/N0)
# -----------------------------------------------------------------------

BITS_PER_SYMBOL = 2   # QPSK
LDPC_CODERATE   = 0.5


def to_eb_n0(snr_db: float, use_ldpc: bool) -> float:
    # config.snr is already Eb/N0 [dB] (passed to sn.utils.ebnodb2no in channel.py)
    # so no conversion is needed.
    # rate = LDPC_CODERATE if use_ldpc else 1.0
    # return snr_db - 10 * np.log10(BITS_PER_SYMBOL * rate)
    return snr_db


def snr_axis_label(eb_n0: bool) -> str:
    return "$E_\mathrm{b}/N_0$ (dB)" if eb_n0 else "$E_s/N_0$ (dB)"
PLOT_CONFIG = {
    "figsize": (9, 6),
    "linewidth": 4,
    "markersize": 12,
    "label_font_size": 24,
    "tick_params_labelsize": 24,
    "legend_font_size": 18,
    "loc": "best",
    "grid": True,
}

# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_results(results_dir: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load every avg_metrics.csv under results_dir and split the rows into
    baseline (use_llm=False) and proposed (use_llm=True).
    """
    csv_files = sorted(results_dir.rglob("avg_metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No avg_metrics.csv found under {results_dir}")

    print(f"Found {len(csv_files)} CSV files")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [skip] {f}: {e}")

    full_df = pd.concat(dfs, ignore_index=True)

    # Fall back to the directory name if the use_llm column is missing
    if "use_llm" not in full_df.columns:
        full_df["use_llm"] = [
            not csv_files[i].parent.name.startswith("baseline")
            for i in range(len(full_df))
        ]
    full_df["use_llm"] = full_df["use_llm"].astype(bool)

    # Fallbacks for snr / temperature / window_size
    for col, key in [("snr", "snr"), ("temperature", "tau"), ("window_size", "win")]:
        if col not in full_df.columns:
            print(f"Warning: '{col}' column missing. Parsing from directory name.")
            full_df[col] = [_parse_from_path(f, key) for f in csv_files[:len(full_df)]]

    for col in ("snr", "temperature", "window_size"):
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

    # use_ldpc: parse from the directory name (ldpctrue/ldpcfalse) if missing
    if "use_ldpc" not in full_df.columns:
        full_df["use_ldpc"] = [_parse_bool_from_path(f, "ldpc") for f in csv_files[:len(full_df)]]
    full_df["use_ldpc"] = full_df["use_ldpc"].astype(bool)

    # num_bits_per_symbol: parse from the directory name (bps2/bps4) if missing, default 2
    if "num_bits_per_symbol" not in full_df.columns:
        full_df["num_bits_per_symbol"] = [_parse_from_path(f, "bps", default=2) for f in csv_files[:len(full_df)]]
    full_df["num_bits_per_symbol"] = pd.to_numeric(full_df["num_bits_per_symbol"], errors="coerce").fillna(2).astype(int)

    df_base = full_df[~full_df["use_llm"]].copy()
    df_prop = full_df[full_df["use_llm"]].copy()

    df_prop = df_prop.dropna(subset=["snr", "temperature", "window_size"])
    df_prop = df_prop.sort_values(["window_size", "snr", "temperature"]).reset_index(drop=True)

    df_base = df_base.dropna(subset=["snr"])
    df_base = df_base.sort_values("snr").reset_index(drop=True)

    print(f"  Proposed : {len(df_prop)} runs")
    print(f"  Baseline : {len(df_base)} runs")
    return df_prop, df_base


def _parse_from_path(csv_path: pathlib.Path, key: str, default: float = float("nan")) -> float:
    name = csv_path.parent.name
    m = re.search(rf"{key}([-\d.]+)", name)
    return float(m.group(1)) if m else default


def _parse_bool_from_path(csv_path: pathlib.Path, key: str, default: bool = False) -> bool:
    name = csv_path.parent.name
    m = re.search(rf"{key}(true|false)", name)
    return m.group(1) == "true" if m else default


# -----------------------------------------------------------------------
# Text summary
# -----------------------------------------------------------------------

def print_summary(df_prop: pd.DataFrame, df_base: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if not df_base.empty:
        print("\n===== Baseline (use_llm=false) =====")
        for metric in METRICS:
            if metric not in df_base.columns:
                continue
            fmt = ".4f" if metric in ("ter", "ber", "lpips", "fid") else ".3f"
            row = df_base[["snr", metric]].set_index("snr")[metric]
            print(f"  {metric.upper():16s}: " + "  ".join(f"SNR={int(s)}→{v:{fmt}}" for s, v in row.items()))

    for win in sorted(df_prop["window_size"].unique()):
        sub = df_prop[df_prop["window_size"] == win]
        print(f"\n===== Proposed | window_size={int(win)} =====")
        for metric in METRICS:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(values=metric, index="snr", columns="temperature", aggfunc="mean")
            fmt = ".4f" if metric in ("ter", "ber", "lpips", "fid") else ".3f"
            print(f"\n  --- {metric.upper()} (SNR × temperature) ---")
            print(pivot.to_string(float_format=lambda x: f"{x:{fmt}}"))


# -----------------------------------------------------------------------
# Legend labels
# -----------------------------------------------------------------------

def make_label(use_llm: bool, use_ldpc: bool, num_bits_per_symbol: int = 2,
               window_size: int | None = None, temperature: float | None = None,
               coderate: float | None = None, show_mod: bool = True) -> str:
    """Build a legend label from the run conditions."""
    mod_suffix = f" ({'16-QAM' if num_bits_per_symbol == 4 else 'QPSK'})" if show_mod else ""
    if use_llm and use_ldpc:
        return f"Proposed+LDPC{mod_suffix}"
    elif use_llm and not use_ldpc:
        return "Proposed"
    elif not use_llm and use_ldpc:
        return f"LDPC{mod_suffix}"
    else:
        return "Uncoded"


# -----------------------------------------------------------------------
# SNR-gain computation and annotation
# -----------------------------------------------------------------------

def _max_snr_gain(bx, by, px, py, m_min: float = -np.inf):
    """Return the maximum SNR gain (dB) at equal metric value.

    Returns (gain_db, snr_at_base, snr_at_prop, metric_at_max) or None.
    gain_db > 0 means the proposed method reaches the same value at a lower SNR.
    m_min: lower bound of the metric values considered (e.g. CLIP Score >= 0.6).
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

    # sort by metric value (np.interp needs increasing xp)
    by_s, bx_s = by[np.argsort(by)], bx[np.argsort(by)]
    py_s, px_s = py[np.argsort(py)], px[np.argsort(py)]

    m_lo = max(by_s[0], py_s[0], m_min)
    m_hi = min(by_s[-1], py_s[-1])
    if m_lo >= m_hi:
        return None

    m_vals = np.linspace(m_lo, m_hi, 500)
    snr_b = np.interp(m_vals, by_s, bx_s)
    snr_p = np.interp(m_vals, py_s, px_s)
    gain  = snr_b - snr_p   # > 0: proposed is better

    idx = int(np.argmax(gain))
    if gain[idx] <= 0:
        return None
    return float(gain[idx]), float(snr_b[idx]), float(snr_p[idx]), float(m_vals[idx])


def _annotate_gain(ax, snr_base: float, snr_prop: float, m_star: float, gain_db: float):
    """Draw a double-headed arrow and the gain text."""
    ax.annotate("", xy=(snr_prop, m_star), xytext=(snr_base, m_star),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text((snr_base + snr_prop) / 2, m_star,
            f"{gain_db:.1f} dB",
            ha="center", va="bottom",
            fontsize=PLOT_CONFIG["legend_font_size"],
            color="black")


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def _save(fig, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")



# baseline vs proposed: metric vs SNR
def plot_vs_snr(df_prop: pd.DataFrame, df_base: pd.DataFrame, out_dir: pathlib.Path,
                eb_n0: bool = False, xlim: tuple | None = None,
                annotate_gain: bool = False, show_mod: bool = True,
                ldpc_bps4_only: bool = True):
    sns.set_style("whitegrid")

    for metric in METRICS:
        if metric not in df_prop.columns:
            continue

        # build the plotting data
        rows = []
        if not df_base.empty and metric in df_base.columns:
            ldpc_vals = sorted(df_base["use_ldpc"].unique())
            bps_vals  = sorted(df_base["num_bits_per_symbol"].unique())
            for ldpc in ldpc_vals:
                for nbps in bps_vals:
                    if ldpc_bps4_only and ldpc and nbps != 4:
                        continue
                    sub = df_base[(df_base["use_ldpc"] == ldpc) & (df_base["num_bits_per_symbol"] == nbps)]
                    if sub.empty:
                        continue
                    label = make_label(use_llm=False, use_ldpc=ldpc, num_bits_per_symbol=nbps, show_mod=show_mod)
                    for _, r in sub.iterrows():
                        x = to_eb_n0(r["snr"], ldpc) if eb_n0 else r["snr"]
                        rows.append({"snr": x, metric: r[metric], "method": label, "use_ldpc": ldpc})
        if not df_prop.empty:
            ldpc_vals = sorted(df_prop["use_ldpc"].unique())
            bps_vals  = sorted(df_prop["num_bits_per_symbol"].unique())
            for win in sorted(df_prop["window_size"].unique()):
                for tau in sorted(df_prop["temperature"].unique()):
                    for ldpc in ldpc_vals:
                        for nbps in bps_vals:
                            if ldpc_bps4_only and ldpc and nbps != 4:
                                continue
                            sub = df_prop[
                                (df_prop["window_size"] == win) &
                                (df_prop["temperature"] == tau) &
                                (df_prop["use_ldpc"] == ldpc) &
                                (df_prop["num_bits_per_symbol"] == nbps)
                            ]
                            if sub.empty:
                                continue
                            label = make_label(use_llm=True, use_ldpc=ldpc, num_bits_per_symbol=nbps,
                                               window_size=win, temperature=tau, show_mod=show_mod)
                            for _, r in sub.iterrows():
                                x = to_eb_n0(r["snr"], ldpc) if eb_n0 else r["snr"]
                                rows.append({"snr": x, metric: r[metric], "method": label, "use_ldpc": ldpc})
        if not rows:
            continue
        plot_df = pd.DataFrame(rows).sort_values("snr")

        ldpc_for_method = plot_df.groupby("method")["use_ldpc"].first().to_dict()
        dashes_map = {m: (5, 2) if v else "" for m, v in ldpc_for_method.items()}

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])
        sns.lineplot(data=plot_df, x="snr", y=metric, hue="method",
                     style="method", markers=True, dashes=dashes_map,
                     linewidth=PLOT_CONFIG["linewidth"],
                     markersize=PLOT_CONFIG["markersize"], ax=ax)

        ax.set_xlabel(snr_axis_label(eb_n0), fontsize=PLOT_CONFIG["label_font_size"])
        ax.set_ylabel(metric_ylabel(metric), fontsize=PLOT_CONFIG["label_font_size"])
        ax.tick_params(labelsize=PLOT_CONFIG["tick_params_labelsize"])
        if xlim is not None:
            ax.set_xlim(xlim)
        if metric in LOG_SCALE_METRICS:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.grid(PLOT_CONFIG["grid"], which="both", alpha=0.3)
        ax.legend(loc=PLOT_CONFIG["loc"], fontsize=PLOT_CONFIG["legend_font_size"],
                  frameon=True, framealpha=0.9)

        # gain annotation: proposed (ldpc=False, bps=2, largest window, tau~1.0) vs uncoded
        if annotate_gain and metric in GAIN_ANNOTATE_METRICS:
            base_sub = df_base[
                (~df_base["use_ldpc"]) & (df_base["num_bits_per_symbol"] == 2)
            ].sort_values("snr")
            prop_cand = df_prop[
                (~df_prop["use_ldpc"]) & (df_prop["num_bits_per_symbol"] == 2)
            ]
            if not base_sub.empty and not prop_cand.empty and metric in base_sub.columns:
                best_win = prop_cand["window_size"].max()
                prop_cand = prop_cand[prop_cand["window_size"] == best_win]
                taus = prop_cand["temperature"].unique()
                best_tau = taus[np.argmin(np.abs(taus - 1.0))]
                prop_sub = prop_cand[prop_cand["temperature"] == best_tau].sort_values("snr")
                bx = np.array([to_eb_n0(v, False) for v in base_sub["snr"]]) if eb_n0 else base_sub["snr"].values
                px = np.array([to_eb_n0(v, False) for v in prop_sub["snr"]]) if eb_n0 else prop_sub["snr"].values
                m_min = 0.6 if metric == "clip_similarity" else -np.inf
                result = _max_snr_gain(bx, base_sub[metric].values, px, prop_sub[metric].values, m_min=m_min)
                if result:
                    _annotate_gain(ax, result[1], result[2], result[3], result[0])

        plt.tight_layout()
        _save(fig, out_dir / f"{metric}_vs_snr.pdf")


# proposed: metric vs window_size (all SNRs in one figure)
def plot_vs_window(df_prop: pd.DataFrame, df_base: pd.DataFrame, out_dir: pathlib.Path,
                   fix_snr=None, eb_n0: bool = False, ldpc_bps4_only: bool = True):
    sns.set_style("whitegrid")
    sub_prop = df_prop[df_prop["snr"] == fix_snr] if fix_snr is not None else df_prop

    for metric in METRICS:
        if metric not in sub_prop.columns:
            continue

        snrs  = sorted(sub_prop["snr"].unique())
        taus  = sorted(sub_prop["temperature"].unique())
        ldpcs = sorted(sub_prop["use_ldpc"].unique())
        bps_vals_prop = sorted(sub_prop["num_bits_per_symbol"].unique())

        # build the proposed rows and the label -> color mapping
        rows = []
        label_meta = []  # (label, color, snr, ldpc, nbps) for the baseline lines
        n_lines = len(snrs) * len(taus) * len(ldpcs) * len(bps_vals_prop)
        palette = sns.color_palette("tab10", n_colors=max(n_lines, 1))
        color_idx = 0

        for snr in snrs:
            for tau in taus:
                for ldpc in ldpcs:
                    for nbps in bps_vals_prop:
                        if ldpc_bps4_only and ldpc and nbps != 4:
                            continue
                        sub = sub_prop[
                            (sub_prop["snr"] == snr) & (sub_prop["temperature"] == tau) &
                            (sub_prop["use_ldpc"] == ldpc) & (sub_prop["num_bits_per_symbol"] == nbps)
                        ]
                        sub = sub.sort_values("window_size")
                        if sub.empty:
                            continue
                        snr_display = to_eb_n0(snr, ldpc)
                        label = f"$E_b/N_0$={round(snr_display):d} dB"
                        color = palette[color_idx]
                        label_meta.append((label, color, snr, ldpc, nbps))
                        color_idx += 1
                        for _, r in sub.iterrows():
                            rows.append({"window_size": r["window_size"], metric: r[metric], "method": label, "use_ldpc": ldpc})

        if not rows:
            continue
        plot_df = pd.DataFrame(rows)
        pal_dict = {lbl: col for lbl, col, _, _, _ in label_meta}
        ldpc_for_method = plot_df.groupby("method")["use_ldpc"].first().to_dict()
        dashes_map = {m: (5, 2) if v else "" for m, v in ldpc_for_method.items()}

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

        # baseline as a horizontal dashed line per SNR (same color as proposed)
        if not df_base.empty and metric in df_base.columns:
            for label, color, snr, ldpc, nbps in label_meta:
                sub_base = df_base[
                    (df_base["snr"] == snr) & (df_base["use_ldpc"] == ldpc) &
                    (df_base["num_bits_per_symbol"] == nbps)
                ]
                if sub_base.empty:
                    continue
                base_val = sub_base[metric]
                if base_val.empty:
                    continue
                snr_display = to_eb_n0(snr, ldpc)
                ax.axhline(base_val.values[0], linestyle="--", color=color,
                           linewidth=PLOT_CONFIG["linewidth"],
                           label=f"$E_b/N_0$={round(snr_display):d} dB (baseline)", zorder=10)

        sns.lineplot(data=plot_df, x="window_size", y=metric, hue="method",
                     style="method", markers=True, dashes=dashes_map,
                     palette=pal_dict, linewidth=PLOT_CONFIG["linewidth"],
                     markersize=PLOT_CONFIG["markersize"], ax=ax)

        ax.set_xlabel("Window size $R$", fontsize=PLOT_CONFIG["label_font_size"])
        ax.set_ylabel(metric_ylabel(metric), fontsize=PLOT_CONFIG["label_font_size"])
        ax.tick_params(labelsize=PLOT_CONFIG["tick_params_labelsize"])
        ax.set_xscale("log", base=2)
        if metric in LOG_SCALE_METRICS:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.grid(PLOT_CONFIG["grid"], which="both", alpha=0.3)
        ax.legend(loc=PLOT_CONFIG["loc"], fontsize=PLOT_CONFIG["legend_font_size"],
                  frameon=True, framealpha=0.9)
        plt.tight_layout()
        _save(fig, out_dir / f"{metric}_vs_window.pdf")


# proposed: metric vs temperature (per SNR)
def plot_vs_tau(df_prop: pd.DataFrame, df_base: pd.DataFrame, out_dir: pathlib.Path,
                fix_window=None, eb_n0: bool = False, log_x: bool = False,
                ldpc_bps4_only: bool = True):
    sns.set_style("whitegrid")
    sub_prop = df_prop[df_prop["window_size"] == fix_window] if fix_window is not None else df_prop

    for metric in METRICS:
        if metric not in sub_prop.columns:
            continue

        rows = []
        ldpcs_tau    = sorted(sub_prop["use_ldpc"].unique())
        bps_vals_tau = sorted(sub_prop["num_bits_per_symbol"].unique())
        for snr in sorted(sub_prop["snr"].unique()):
            for ldpc in ldpcs_tau:
                for nbps in bps_vals_tau:
                    if ldpc_bps4_only and ldpc and nbps != 4:
                        continue
                    sub = sub_prop[
                        (sub_prop["snr"] == snr) & (sub_prop["use_ldpc"] == ldpc) &
                        (sub_prop["num_bits_per_symbol"] == nbps)
                    ]
                    sub = sub.sort_values("temperature")
                    if sub.empty:
                        continue
                    snr_display = to_eb_n0(snr, ldpc)
                    label = f"$E_b/N_0$={round(snr_display):d} dB"
                    for _, r in sub.iterrows():
                        rows.append({"tau": r["temperature"], metric: r[metric], "method": label, "use_ldpc": ldpc})
        if not rows:
            continue
        plot_df = pd.DataFrame(rows).sort_values("tau")

        ldpc_for_method = plot_df.groupby("method")["use_ldpc"].first().to_dict()
        dashes_map = {m: (5, 2) if v else "" for m, v in ldpc_for_method.items()}

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])
        sns.lineplot(data=plot_df, x="tau", y=metric, hue="method",
                     style="method", markers=True, dashes=dashes_map,
                     linewidth=PLOT_CONFIG["linewidth"],
                     markersize=PLOT_CONFIG["markersize"], ax=ax)

        ax.set_xlabel(r"Temperature $\tau$", fontsize=PLOT_CONFIG["label_font_size"])
        ax.set_ylabel(metric_ylabel(metric), fontsize=PLOT_CONFIG["label_font_size"])
        ax.tick_params(labelsize=PLOT_CONFIG["tick_params_labelsize"])
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        if metric in LOG_SCALE_METRICS:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.grid(PLOT_CONFIG["grid"], which="both", alpha=0.3)
        ax.legend(loc=PLOT_CONFIG["loc"], fontsize=PLOT_CONFIG["legend_font_size"],
                  frameon=True, framealpha=0.9)
        plt.tight_layout()
        _save(fig, out_dir / f"{metric}_vs_tau.pdf")


# -----------------------------------------------------------------------
# Token-Level Transition Analysis
# -----------------------------------------------------------------------

NORM_TOKENS = 4096  # normalize to 4096 tokens per image as in the paper


def load_token_transition(results_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load token_correct.npy of every run, pair baseline and proposed runs and
    return the token transition counts.

    Columns of the returned DataFrame:
        snr, use_ldpc,
        base_acc, prop_acc,
        stayed, successful_correction, miscorrection, uncorrected
    (all in [%], normalized to NORM_TOKENS tokens)
    """
    npy_files = sorted(results_dir.rglob("token_correct.npy"))
    if not npy_files:
        return pd.DataFrame()

    records = []
    for f in npy_files:
        # parse the run conditions from the directory name
        parent = f.parent.name
        snr_m = re.search(r"snr([-\d]+)", parent)
        ldpc_m = re.search(r"ldpc(true|false)", parent)
        llm_m  = re.search(r"llm(true|false)", parent)
        if not snr_m:
            continue
        snr    = int(snr_m.group(1))
        use_ldpc = ldpc_m and ldpc_m.group(1) == "true"
        use_llm  = llm_m  and llm_m.group(1) == "true"
        correct = np.load(f)
        records.append({"snr": snr, "use_ldpc": use_ldpc, "use_llm": use_llm,
                         "correct": correct, "path": f})

    if not records:
        return pd.DataFrame()

    rows = []
    base_records = [r for r in records if not r["use_llm"]]
    prop_records = [r for r in records if r["use_llm"]]

    for base in base_records:
        for prop in prop_records:
            if base["snr"] != prop["snr"] or base["use_ldpc"] != prop["use_ldpc"]:
                continue
            b = base["correct"]
            p = prop["correct"]
            L = min(len(b), len(p))
            b, p = b[:L], p[:L]
            rows.append({
                "snr":                  base["snr"],
                "use_ldpc":             base["use_ldpc"],
                "base_acc":             b.mean() * 100,
                "prop_acc":             p.mean() * 100,
                "stayed":               (b & p).mean() * 100,
                "successful_correction": (~b & p).mean() * 100,
                "miscorrection":        (b & ~p).mean() * 100,
                "uncorrected":          (~b & ~p).mean() * 100,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["use_ldpc", "snr"]).reset_index(drop=True)
    return df


def print_transition_table(df: pd.DataFrame):
    if df.empty:
        print("\n[Transition] no token_correct.npy found; skipped.")
        return
    print("\n" + "=" * 90)
    print(f"  TOKEN-LEVEL TRANSITION ANALYSIS (normalized per {NORM_TOKENS} tokens)")
    print("=" * 90)
    for ldpc in sorted(df["use_ldpc"].unique()):
        sub = df[df["use_ldpc"] == ldpc]
        tag = "LDPC" if ldpc else "Uncoded"
        print(f"\n  [{tag}]")
        print(f"  {'SNR':>5} {'Baseline':>10} {'Proposed':>10} {'Stayed':>8} "
              f"{'Succ.Corr':>11} {'Misc.Corr':>11} {'Uncorr':>8}")
        for _, r in sub.iterrows():
            print(f"  {int(r['snr']):>5} {r['base_acc']:>9.1f}% {r['prop_acc']:>9.1f}% "
                  f"{r['stayed']:>7.1f}% {r['successful_correction']:>10.1f}% "
                  f"{r['miscorrection']:>10.1f}% {r['uncorrected']:>7.1f}%")


def plot_token_transition(df: pd.DataFrame, out_dir: pathlib.Path, eb_n0: bool = False):
    if df.empty:
        return
    sns.set_style("whitegrid")

    for ldpc in sorted(df["use_ldpc"].unique()):
        sub = df[df["use_ldpc"] == ldpc].sort_values("snr")
        tag = "ldpc" if ldpc else "uncoded"

        snrs = np.array([to_eb_n0(s, ldpc) for s in sub["snr"].values]) if eb_n0 else sub["snr"].values
        stayed   = sub["stayed"].values
        succ     = sub["successful_correction"].values
        misc     = sub["miscorrection"].values
        uncorr   = sub["uncorrected"].values

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])
        bar_w = 0.6
        x = np.arange(len(snrs))

        ax.bar(x, stayed,  bar_w, label="Stayed correct",          color="#4C72B0")
        ax.bar(x, succ,    bar_w, bottom=stayed,                   label="Successful correction", color="#55A868")
        ax.bar(x, misc,    bar_w, bottom=stayed + succ,            label="Miscorrection",         color="#C44E52")
        ax.bar(x, uncorr,  bar_w, bottom=stayed + succ + misc,     label="Uncorrected",           color="#B0B0B0")

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in snrs])
        ax.set_xlabel(snr_axis_label(eb_n0), fontsize=PLOT_CONFIG["label_font_size"])
        ax.set_ylabel(f"Tokens [%] (norm. {NORM_TOKENS})", fontsize=PLOT_CONFIG["label_font_size"])
        ax.tick_params(labelsize=PLOT_CONFIG["tick_params_labelsize"])
        ax.legend(loc="upper left", fontsize=PLOT_CONFIG["legend_font_size"],
                  frameon=True, framealpha=0.9)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        _save(fig, out_dir / f"token_transition_{tag}.pdf")

        # successful correction / miscorrection curves
        fig2, ax2 = plt.subplots(figsize=PLOT_CONFIG["figsize"])
        ax2.plot(snrs, succ, marker="o", linewidth=PLOT_CONFIG["linewidth"],
                 markersize=PLOT_CONFIG["markersize"],
                 label="Successful correction", color="#55A868")
        ax2.plot(snrs, misc, marker="s", linewidth=PLOT_CONFIG["linewidth"],
                 markersize=PLOT_CONFIG["markersize"],
                 label="Miscorrection",         color="#C44E52", linestyle="--")
        ax2.set_xlabel(snr_axis_label(eb_n0), fontsize=PLOT_CONFIG["label_font_size"])
        ax2.set_ylabel(f"Tokens [%] (norm. {NORM_TOKENS})", fontsize=PLOT_CONFIG["label_font_size"])
        ax2.tick_params(labelsize=PLOT_CONFIG["tick_params_labelsize"])
        ax2.legend(loc="best", fontsize=PLOT_CONFIG["legend_font_size"])
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig2, out_dir / f"correction_vs_snr_{tag}.pdf")


# -----------------------------------------------------------------------
# main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs", help="sweep root directory")
    parser.add_argument("--out_dir", default=None, help="output directory (default: results_dir/analysis)")
    parser.add_argument("--fix_window", type=float, default=None, help="fix window_size and compare SNR x temperature")
    parser.add_argument("--fix_snr", type=float, default=None, help="fix the SNR and compare window sizes")
    parser.add_argument("--eb_n0", action="store_true",
                        help="use Eb/N0 (per information bit) on the SNR axis")
    parser.add_argument("--tau_log_x", action="store_true",
                        help="log-scale tau axis in the vs_tau plots")
    parser.add_argument("--snr_xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="x range of the vs_snr plots (e.g. --snr_xlim -6 6)")
    parser.add_argument("--annotate_gain", action="store_true",
                        help="annotate the maximum SNR gain in the PSNR/LPIPS/CLIP Score vs SNR plots")
    parser.add_argument("--simple_labels", action="store_true",
                        help="omit the modulation (QPSK)/(16-QAM) from legend labels")
    parser.add_argument("--plot_all", action="store_true",
                        help="plot every bps for LDPC (default: bps4 only)")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else results_dir / "analysis"

    df_prop, df_base = load_results(results_dir)

    if not df_prop.empty:
        print(f"\nSNR range (proposed) : {sorted(df_prop['snr'].unique())}")
        print(f"Temperature range    : {sorted(df_prop['temperature'].unique())}")
        print(f"Window size range    : {sorted(df_prop['window_size'].unique())}")
    if not df_base.empty:
        print(f"SNR range (baseline) : {sorted(df_base['snr'].unique())}")

    print_summary(df_prop, df_base)

    df_prop_filtered = df_prop[df_prop["window_size"] == args.fix_window] if args.fix_window else df_prop
    ldpc_bps4_only = not args.plot_all
    show_mod = args.plot_all and not args.simple_labels

    xlim = tuple(args.snr_xlim) if args.snr_xlim else None
    plot_vs_snr(df_prop_filtered, df_base, out_dir / "vs_snr", eb_n0=args.eb_n0, xlim=xlim,
                annotate_gain=args.annotate_gain, show_mod=show_mod,
                ldpc_bps4_only=ldpc_bps4_only)
    plot_vs_window(df_prop, df_base, out_dir / "vs_window", fix_snr=args.fix_snr, eb_n0=args.eb_n0,
                   ldpc_bps4_only=ldpc_bps4_only)
    plot_vs_tau(df_prop, df_base, out_dir / "vs_tau", fix_window=args.fix_window, eb_n0=args.eb_n0,
                log_x=args.tau_log_x, ldpc_bps4_only=ldpc_bps4_only)

    df_trans = load_token_transition(results_dir)
    print_transition_table(df_trans)
    plot_token_transition(df_trans, out_dir / "transition", eb_n0=args.eb_n0)

    csv_path = out_dir / "all_results.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([df_prop, df_base], ignore_index=True).to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")


if __name__ == "__main__":
    main()
