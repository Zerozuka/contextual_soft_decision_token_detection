#!/usr/bin/env zsh
# sweep.sh
# Sweep over SNR / temperature / window_size / use_llm / use_ldpc.
#
# Usage (from the repository root):
#   bash scripts/sweep.sh [OPTIONS]
#
# Options:
#   --snr_list        "space-separated SNRs"  (default: "-2 -1 0 1 2 3 4 5 6 7 8")
#   --use_llm_list    "space-separated true|false"  (default: "true")
#   --use_ldpc_list   "space-separated true|false"  (default: "false")
#   --temperature_list "space-separated values"    (default: "1.0")
#   --window_size_list "space-separated values"    (default: "4096")  ignored when use_llm=false
#   --ldpc_n          int                      (default: 2048)
#   --ldpc_coderate   float                    (default: 0.5)
#   --ldpc_num_iter   int                      (default: 20)
#   --num_images      int                      (default: 5)
#   --batch_size      int                      (default: 5)
#   --beam_width      int                      (default: 1)
#   --tag             str                      (default: "")  suffix of the output directory
#   --output_dir      str                      (default: "") append to an existing directory; no timestamp is generated and finished runs are skipped
#
# Examples:
#   # LLM, No LDPC, SNR=-2~8, temperature=1.0, window=4096
#   bash scripts/sweep.sh
#
#   # hard decision / LMM x with/without LDPC, all combinations
#   bash scripts/sweep.sh --use_llm_list "false true" --use_ldpc_list "false true"
#
#   # LMM with and without LDPC
#   bash scripts/sweep.sh --use_ldpc_list "false true"
#
#   # grid search over temperature x window_size
#   bash scripts/sweep.sh --temperature_list "0.5 1.0 5.0" --window_size_list "512 4096"
#
#   # fine sweep with 0.5 dB steps
#   bash scripts/sweep.sh --snr_list "-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0"
# 

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# activate the venv
SCRIPT_DIR="${0:A:h}"
VENV="${SCRIPT_DIR}/../venv/bin/activate"
if [[ -f "${VENV}" ]]; then
    source "${VENV}"
fi

# ---- defaults ----
SNR_LIST="-3 -2 -1 0 1 2 3 4 5 6 7 8 9"
USE_LLM_LIST="false true"
USE_LDPC_LIST="false"
TEMPERATURE_LIST="1.0"
WINDOW_SIZE_LIST="4096"
LDPC_N=2048
LDPC_CODERATE=0.5
LDPC_NUM_ITER=20
NUM_IMAGES=100
BATCH_SIZE=5
BEAM_WIDTH=1
TAG=""
OUTPUT_DIR=""

# ---- parse options ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --snr_list)         SNR_LIST="$2";         shift 2 ;;
        --use_llm_list)     USE_LLM_LIST="$2";      shift 2 ;;
        --use_ldpc_list)    USE_LDPC_LIST="$2";     shift 2 ;;
        --temperature_list) TEMPERATURE_LIST="$2";  shift 2 ;;
        --window_size_list) WINDOW_SIZE_LIST="$2";  shift 2 ;;
        --ldpc_n)           LDPC_N="$2";            shift 2 ;;
        --ldpc_coderate)    LDPC_CODERATE="$2";     shift 2 ;;
        --ldpc_num_iter)    LDPC_NUM_ITER="$2";     shift 2 ;;
        --num_images)       NUM_IMAGES="$2";        shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";        shift 2 ;;
        --beam_width)       BEAM_WIDTH="$2";        shift 2 ;;
        --tag)              TAG="_$2";              shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2";        shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- output directory ----
if [[ -n "${OUTPUT_DIR}" ]]; then
    SWEEP_DIR="${OUTPUT_DIR}"
    mkdir -p "${SWEEP_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SWEEP_DIR="outputs/sweep_${TIMESTAMP}${TAG}"
    mkdir -p "${SWEEP_DIR}"
fi

echo "=== Sweep start: $(date) ==="
echo "Sweep dir        : ${SWEEP_DIR}"
echo "SNR list         : ${SNR_LIST}"
echo "use_llm list     : ${USE_LLM_LIST}"
echo "use_ldpc list    : ${USE_LDPC_LIST}"
echo "temperature list : ${TEMPERATURE_LIST}"
echo "window_size list : ${WINDOW_SIZE_LIST}  (ignored when use_llm=false)"
echo "ldpc_n           : ${LDPC_N}, coderate=${LDPC_CODERATE}, num_iter=${LDPC_NUM_ITER}"
echo "num_images       : ${NUM_IMAGES}, batch_size=${BATCH_SIZE}, beam_width=${BEAM_WIDTH}"
echo ""

# record the settings
cat > "${SWEEP_DIR}/sweep_config.txt" << EOF
timestamp:        $(date)
tag:              ${TAG}
snr_list:         ${SNR_LIST}
use_llm_list:     ${USE_LLM_LIST}
use_ldpc_list:    ${USE_LDPC_LIST}
temperature_list: ${TEMPERATURE_LIST}
window_size_list: ${WINDOW_SIZE_LIST}
ldpc_n:           ${LDPC_N}
ldpc_coderate:    ${LDPC_CODERATE}
ldpc_num_iter:    ${LDPC_NUM_ITER}
num_images:       ${NUM_IMAGES}
batch_size:       ${BATCH_SIZE}
beam_width:       ${BEAM_WIDTH}
EOF

# ---- main loop ----
SNR_ARR=(${=SNR_LIST})
USE_LLM_ARR=(${=USE_LLM_LIST})
USE_LDPC_ARR=(${=USE_LDPC_LIST})
TEMPERATURE_ARR=(${=TEMPERATURE_LIST})
WINDOW_SIZE_ARR=(${=WINDOW_SIZE_LIST})

for USE_LLM in "${USE_LLM_ARR[@]}"; do
    for USE_LDPC in "${USE_LDPC_ARR[@]}"; do
        LDPC_ARGS=()
        if [[ "${USE_LDPC}" == "true" ]]; then
            LDPC_ARGS=("channel.ldpc_n=${LDPC_N}" "channel.ldpc_coderate=${LDPC_CODERATE}" "channel.ldpc_num_iter=${LDPC_NUM_ITER}")
        fi

        for SNR in "${SNR_ARR[@]}"; do
            if [[ "${USE_LLM}" == "false" ]]; then
                RUN_NAME="snr${SNR}_llm${USE_LLM}_ldpc${USE_LDPC}"
                RUN_DIR="${SWEEP_DIR}/${RUN_NAME}"
                echo "--- [$(date +%H:%M:%S)] snr=${SNR}, llm=${USE_LLM}, ldpc=${USE_LDPC} ---"

                python src/main.py \
                    channel.snr="${SNR}" \
                    channel.use_ldpc="${USE_LDPC}" \
                    "${LDPC_ARGS[@]}" \
                    receiver.use_llm="${USE_LLM}" \
                    num_images="${NUM_IMAGES}" \
                    dataset.batch_size="${BATCH_SIZE}" \
                    hydra.run.dir="${RUN_DIR}" \
                    2>&1 | tee -a "${SWEEP_DIR}/sweep.log"
            else
                for TEMP in "${TEMPERATURE_ARR[@]}"; do
                    for WIN in "${WINDOW_SIZE_ARR[@]}"; do
                        RUN_NAME="snr${SNR}_llm${USE_LLM}_tau${TEMP}_win${WIN}_ldpc${USE_LDPC}"
                        RUN_DIR="${SWEEP_DIR}/${RUN_NAME}"
                        echo "--- [$(date +%H:%M:%S)] snr=${SNR}, llm=${USE_LLM}, tau=${TEMP}, win=${WIN}, ldpc=${USE_LDPC} ---"

                        python src/main.py \
                            channel.snr="${SNR}" \
                            channel.use_ldpc="${USE_LDPC}" \
                            "${LDPC_ARGS[@]}" \
                            receiver.use_llm="${USE_LLM}" \
                            receiver.temperature="${TEMP}" \
                            receiver.beam_width="${BEAM_WIDTH}" \
                            emu3.window_size="${WIN}" \
                            num_images="${NUM_IMAGES}" \
                            dataset.batch_size="${BATCH_SIZE}" \
                            hydra.run.dir="${RUN_DIR}" \
                            2>&1 | tee -a "${SWEEP_DIR}/sweep.log"
                    done
                done
            fi
        done
    done
done

echo ""
echo "=== Done: $(date) ==="
echo "Results : ${SWEEP_DIR}"
echo "Analyze : python scripts/analyze_sweep.py --results_dir ${SWEEP_DIR}"
