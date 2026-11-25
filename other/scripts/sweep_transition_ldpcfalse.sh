#!/usr/bin/env zsh
# sweep_transition_ldpcfalse.sh - rerun for the transition analysis (ldpc=false)
# llm=false + llm=true, ldpc=false, SNR=-2~8 (Eb/N0=-5~5 dB)

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="${0:A:h}"
VENV="${SCRIPT_DIR}/../venv/bin/activate"
if [[ -f "${VENV}" ]]; then source "${VENV}"; fi

SWEEP_DIR="outputs/sweep_20260417_203949_transition_ldpcfalse"
mkdir -p "${SWEEP_DIR}"

NUM_IMAGES=100
BATCH_SIZE=5
TEMP=1.0
WIN=4096

echo "=== Sweep start: $(date) ==="
echo "Sweep dir: ${SWEEP_DIR}"

run() {
    local SNR=$1 LLM=$2
    local RUN_NAME="snr${SNR}_llm${LLM}_tau${TEMP}_win${WIN}_ldpcfalse"
    [[ "${LLM}" == "false" ]] && RUN_NAME="snr${SNR}_llmfalse_ldpcfalse"
    echo "--- [$(date +%H:%M:%S)] snr=${SNR}, llm=${LLM}, ldpc=false ---"
    python src/main.py \
        channel.snr="${SNR}" \
        channel.use_ldpc=false \
        receiver.use_llm="${LLM}" \
        receiver.temperature="${TEMP}" \
        emu3.window_size="${WIN}" \
        num_images="${NUM_IMAGES}" \
        dataset.batch_size="${BATCH_SIZE}" \
        hydra.run.dir="${SWEEP_DIR}/${RUN_NAME}" \
        2>&1 | tee -a "${SWEEP_DIR}/sweep.log"
}

for SNR in 0 1 2 3 4 5 6 7 8; do
    run ${SNR} false
    run ${SNR} true
done

echo ""
echo "=== Done: $(date) ==="
echo "Results: ${SWEEP_DIR}"
