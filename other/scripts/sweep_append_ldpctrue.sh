#!/usr/bin/env zsh
# append the ldpc=true runs to outputs/sweep_20260422_015717_
# SNR=-6~6, llm=false+true

SCRIPT_DIR="${0:A:h}"

zsh "${SCRIPT_DIR}/sweep.sh" \
    --output_dir "outputs/sweep_20260422_015717_" \
    --snr_list "-6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6" \
    --use_llm_list "false true" \
    --use_ldpc_list "true" \
    --num_images 100 \
    --batch_size 5
