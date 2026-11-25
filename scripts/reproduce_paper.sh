#!/usr/bin/env bash
# Reproduce the figures and the table of
# "Contextual Soft-Decision Token Detection via LMM-Based Generative Semantic Prior".
#
# Usage (from the repository root):
#   bash scripts/reproduce_paper.sh figures                       # figures/table from the recorded results (no GPU, < 1 min)
#   bash scripts/reproduce_paper.sh run [SWEEP...]                # rerun the experiments with the recorded settings (GPU, long)
#   bash scripts/reproduce_paper.sh figures outputs/paper_repro   # figures/table from your own rerun
#
# SWEEP is a directory name under results/sweeps (default: all three):
#   sweep_20260413_095050_            figures: TER / PSNR / LPIPS / CLIP Score vs Eb/N0
#   sweep_20260414_174518__tau_sweep  figure:  TER vs tau
#   sweep_20260416_133456__window     table:   TER vs window size W
#
# Environment variables:
#   PYTHON        Python interpreter (default: python)
#   DATASET_PATH  path to FFHQ 512x512; overrides the recorded dataset.path
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RECORDED="results/sweeps"
ALL_SWEEPS=(sweep_20260413_095050_ sweep_20260414_174518__tau_sweep sweep_20260416_133456__window)

mode="${1:-figures}"
shift || true

case "${mode}" in
figures)
    src="${1:-${RECORDED}}"
    out="outputs/paper_figures/$(basename "${src}")"   # outputs/ is git-ignored
    # Options used for the paper figures (outputs are pixel-identical to results/figures)
    "${PYTHON}" scripts/analyze_sweep.py --results_dir "${src}/sweep_20260413_095050_" \
        --out_dir "${out}/snr" --eb_n0 --snr_xlim -3 8 --annotate_gain
    "${PYTHON}" scripts/analyze_sweep.py --results_dir "${src}/sweep_20260414_174518__tau_sweep" \
        --out_dir "${out}/tau" --eb_n0
    "${PYTHON}" scripts/paper_window_table.py "${src}/sweep_20260416_133456__window"
    echo
    echo "Figures: ${out}/snr/vs_snr/{ter,psnr,lpips,clip_similarity}_vs_snr.pdf"
    echo "         ${out}/tau/vs_tau/ter_vs_tau.pdf"
    echo "Reference figures: results/figures/"
    ;;
run)
    sweeps=("$@")
    [[ ${#sweeps[@]} -eq 0 ]] && sweeps=("${ALL_SWEEPS[@]}")
    for sweep in "${sweeps[@]}"; do
        for ov in "${RECORDED}/${sweep}"/*/overrides.yaml; do
            run="$(basename "$(dirname "${ov}")")"
            run_dir="outputs/paper_repro/${sweep}/${run}"
            if [[ -f "${run_dir}/avg_metrics.csv" ]]; then
                echo "skip (done): ${run_dir}"
                continue
            fi
            # overrides.yaml is the Hydra override list ("- key=value")
            args=()
            while IFS= read -r a; do
                if [[ -n "${DATASET_PATH:-}" && "${a}" == dataset.path=* ]]; then
                    continue
                fi
                args+=("${a}")
            done < <(sed -n 's/^- //p' "${ov}")
            if [[ -n "${DATASET_PATH:-}" ]]; then
                args+=("dataset.path=${DATASET_PATH}")
            fi
            echo "--- [$(date +%H:%M:%S)] ${sweep}/${run}"
            "${PYTHON}" src/main.py "${args[@]}" hydra.run.dir="${run_dir}"
        done
    done
    echo "Done. Figures: bash scripts/reproduce_paper.sh figures outputs/paper_repro"
    ;;
*)
    echo "usage: $0 {figures [RESULTS_ROOT] | run [SWEEP...]}" >&2
    exit 1
    ;;
esac
