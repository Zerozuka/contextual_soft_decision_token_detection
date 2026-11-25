#!/usr/bin/env zsh
# sweep_missing_llm.sh - 5 additional runs: missing SNRs with llm=true
# ldpc=false, llm=true: SNR=-3 (Eb/N0=-6 dB)
# ldpc=true,  llm=true: SNR=-6,-5,-4,-3 (Eb/N0=-6~-3 dB)

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="${0:A:h}"
VENV="${SCRIPT_DIR}/../venv/bin/activate"
if [[ -f "${VENV}" ]]; then source "${VENV}"; fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="outputs/sweep_${TIMESTAMP}_llm_missing"
mkdir -p "${SWEEP_DIR}"

NUM_IMAGES=100
BATCH_SIZE=5
TEMP=1.0
WIN=4096

echo "=== Sweep start: $(date) ==="
echo "Sweep dir: ${SWEEP_DIR}"

run_llm() {
    local SNR=$1 LDPC=$2
    local LDPC_ARGS=()
    if [[ "${LDPC}" == "true" ]]; then
        LDPC_ARGS=("channel.ldpc_n=2048" "channel.ldpc_coderate=0.5" "channel.ldpc_num_iter=20")
    fi
    local RUN_NAME="snr${SNR}_llmtrue_tau${TEMP}_win${WIN}_ldpc${LDPC}"
    echo "--- [$(date +%H:%M:%S)] snr=${SNR}, ldpc=${LDPC} ---"
    python src/main.py \
        channel.snr="${SNR}" \
        channel.use_ldpc="${LDPC}" \
        "${LDPC_ARGS[@]}" \
        receiver.use_llm=true \
        receiver.temperature="${TEMP}" \
        emu3.window_size="${WIN}" \
        num_images="${NUM_IMAGES}" \
        dataset.batch_size="${BATCH_SIZE}" \
        hydra.run.dir="${SWEEP_DIR}/${RUN_NAME}" \
        2>&1 | tee -a "${SWEEP_DIR}/sweep.log"
}

run_llm -3 false
run_llm -6 true
run_llm -5 true
run_llm -4 true
run_llm -3 true

echo ""
echo "=== Done: $(date) ==="
echo "Results: ${SWEEP_DIR}"
