#!/usr/bin/env zsh
# sweep_missing_baseline.sh - missing runs with use_llm=false
# ldpc=false, llm=false: all SNR=-3..9 (Eb/N0=-6..6 dB)
# ldpc=true,  llm=false: only SNR=-6,-5,-4,-3 (Eb/N0=-6..-3 dB)

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="${0:A:h}"
VENV="${SCRIPT_DIR}/../venv/bin/activate"
if [[ -f "${VENV}" ]]; then source "${VENV}"; fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="outputs/sweep_${TIMESTAMP}_baseline_missing"
mkdir -p "${SWEEP_DIR}"

NUM_IMAGES=100
BATCH_SIZE=5

echo "=== Sweep start: $(date) ==="
echo "Sweep dir: ${SWEEP_DIR}"

run_baseline() {
    local SNR=$1 LDPC=$2
    local LDPC_ARGS=()
    if [[ "${LDPC}" == "true" ]]; then
        LDPC_ARGS=("channel.ldpc_n=2048" "channel.ldpc_coderate=0.5" "channel.ldpc_num_iter=20")
    fi
    local RUN_NAME="snr${SNR}_llmfalse_ldpc${LDPC}"
    echo "--- [$(date +%H:%M:%S)] snr=${SNR}, ldpc=${LDPC} ---"
    python src/main.py \
        channel.snr="${SNR}" \
        channel.use_ldpc="${LDPC}" \
        "${LDPC_ARGS[@]}" \
        receiver.use_llm=false \
        num_images="${NUM_IMAGES}" \
        dataset.batch_size="${BATCH_SIZE}" \
        hydra.run.dir="${SWEEP_DIR}/${RUN_NAME}" \
        2>&1 | tee -a "${SWEEP_DIR}/sweep.log"
}

# ldpc=false: SNR -3~9 (Eb/N0 -6~6 dB)
for SNR in -3 -2 -1 0 1 2 3 4 5 6 7 8 9; do
    run_baseline ${SNR} false
done

# ldpc=true: only SNR -6..-3 (Eb/N0 -6..-3 dB)
for SNR in -6 -5 -4 -3; do
    run_baseline ${SNR} true
done

echo ""
echo "=== Done: $(date) ==="
echo "Results: ${SWEEP_DIR}"
