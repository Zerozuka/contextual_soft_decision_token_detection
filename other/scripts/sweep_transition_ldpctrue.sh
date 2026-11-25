#!/usr/bin/env zsh
# sweep_transition_ldpctrue.sh - rerun for the transition analysis (ldpc=true)
# llm=false + llm=true, ldpc=true, SNR=-2~6 (Eb/N0=-2~6 dB)

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="${0:A:h}"
VENV="${SCRIPT_DIR}/../venv/bin/activate"
if [[ -f "${VENV}" ]]; then source "${VENV}"; fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="outputs/sweep_${TIMESTAMP}_transition_ldpctrue"
mkdir -p "${SWEEP_DIR}"

NUM_IMAGES=100
BATCH_SIZE=5
TEMP=1.0
WIN=4096
LDPC_ARGS=(channel.ldpc_n=2048 channel.ldpc_coderate=0.5 channel.ldpc_num_iter=20)

echo "=== Sweep start: $(date) ==="
echo "Sweep dir: ${SWEEP_DIR}"

run() {
    local SNR=$1 LLM=$2
    local RUN_NAME="snr${SNR}_llm${LLM}_tau${TEMP}_win${WIN}_ldpctrue"
    [[ "${LLM}" == "false" ]] && RUN_NAME="snr${SNR}_llmfalse_ldpctrue"
    echo "--- [$(date +%H:%M:%S)] snr=${SNR}, llm=${LLM}, ldpc=true ---"
    python src/main.py \
        channel.snr="${SNR}" \
        channel.use_ldpc=true \
        "${LDPC_ARGS[@]}" \
        receiver.use_llm="${LLM}" \
        receiver.temperature="${TEMP}" \
        emu3.window_size="${WIN}" \
        num_images="${NUM_IMAGES}" \
        dataset.batch_size="${BATCH_SIZE}" \
        hydra.run.dir="${SWEEP_DIR}/${RUN_NAME}" \
        2>&1 | tee -a "${SWEEP_DIR}/sweep.log"
}

for SNR in -2 -1 0 1 2 3 4 5 6; do
    run ${SNR} false
    run ${SNR} true
done

echo ""
echo "=== Done: $(date) ==="
echo "Results: ${SWEEP_DIR}"
