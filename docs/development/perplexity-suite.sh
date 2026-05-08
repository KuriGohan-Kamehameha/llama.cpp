#!/bin/bash
# Per-quant perplexity gate for the ESIMD opt-in path.
#
# Validates that ESIMD generation quality is statistically indistinguishable
# from the vanilla SYCL kernel on each of the 5 quants in this fork's
# coverage. Q4_K already verified (commit 1a3014f); this script runs the
# same check for Q4_0, Q5_K, Q6_K, Q8_0.
#
# Reproduction:
#   1. Have the fork built with -DGGML_SYCL_ESIMD=ON (per docs/development/SYCL_ESIMD_PERF.md).
#   2. Have wiki.test.raw available (https://huggingface.co/datasets/Salesforce/wikitext).
#   3. Have one model file per quant accessible via $MODEL_<quant> env var.
#   4. Set $LLAMA_PERPLEXITY to the perplexity binary path.
#   5. Run: ./perplexity-suite.sh
#
# On Xe-LPG:  use -fa off  (the auto path triggers UR_RESULT_ERROR_DEVICE_LOST).

set -eo pipefail

: "${LLAMA_PERPLEXITY:?need llama-perplexity path}"
: "${WIKI:?need wiki.test.raw path}"
: "${MODEL_Q4_0:?need Q4_0 GGUF}"
: "${MODEL_Q5_K:?need Q5_K_M GGUF}"
: "${MODEL_Q6_K:?need Q6_K GGUF}"
: "${MODEL_Q8_0:?need Q8_0 GGUF}"

CHUNKS="${CHUNKS:-50}"
NGL="${NGL:-999}"
RUNS="${RUNS:-3}"

run_ppl() {
    local label="$1"
    local model="$2"
    local esimd_env="$3"
    local i

    for i in $(seq 1 "$RUNS"); do
        printf '\n=== %s run %d/%d ===\n' "$label" "$i" "$RUNS"
        env $esimd_env ZES_ENABLE_SYSMAN=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
            "$LLAMA_PERPLEXITY" \
            -m "$model" -f "$WIKI" -ngl "$NGL" --chunks "$CHUNKS" -fa off 2>&1 \
            | tee "/tmp/ppl-${label// /-}-run${i}.log" \
            | grep -E "^\[1\]|^Final estimate" || true
    done
}

for q in Q4_0 Q5_K Q6_K Q8_0; do
    model_var="MODEL_${q}"
    model="${!model_var}"
    echo
    echo "############################################################"
    echo "## $q  model: $model"
    echo "############################################################"
    run_ppl "${q}-vanilla" "$model" ""
    run_ppl "${q}-esimd"   "$model" "GGML_SYCL_USE_ESIMD=1"
done

echo
echo "All runs complete. Compare per-quant final-PPL between vanilla"
echo "and esimd runs. Acceptance: |Δppl| within ~2σ of within-path"
echo "stddev, ideally < 0.5% relative. Per-chunk PPL stream sha256"
echo "match is the strongest signal (Q4_K precedent: identical sha256)."
