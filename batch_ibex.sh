#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HWE-Bench Ibex batch run
# Assumption:
#   - You are already in the hwe-bench repo root.
#   - DASHSCOPE_API_KEY has already been exported.
#   - hf dataset has already been downloaded to datasets/.
# ============================================================

JOB_NAME="ibex-qwen3.6"
DATASET_FILE="datasets/lowRISC__ibex.jsonl"
TASK_DIR="tasks/hwe-bench-ibex"
RESULT_DIR="results/${JOB_NAME}"
JOB_DIR="jobs/${JOB_NAME}"

# Harbor agent concurrency.
# For your 12 logical-core machine, start with 4.
# You can try 6 later.
N_CONCURRENT=4

# Evaluator concurrency.
# This controls how many PRs/cases are evaluated in parallel.
# Start with 4; try 6 only if CPU/disk are stable.
EVAL_WORKERS=4

# ============================================================
# 1. Pull Ibex images
# ============================================================

./scripts/pull_images.sh ibex

# ============================================================
# 2. Generate task directories for all Ibex PRs
# ============================================================

rm -rf "${TASK_DIR}"

uv run python -m hwe_bench.harness.harbor.adapter \
  --input "${DATASET_FILE}" \
  --output "${TASK_DIR}/"

# ============================================================
# 3. Run the agent on all Ibex PRs
# ============================================================

harbor run --path "${TASK_DIR}/" \
  -a openhands-sdk \
  -m openai/qwen3.6-plus \
  --ae LLM_API_KEY="${DASHSCOPE_API_KEY}" \
  --ae LLM_BASE_URL="https://coding.dashscope.aliyuncs.com/v1" \
  --ae MAX_ITERATIONS=500 \
  -k 1 \
  -r 2 \
  --n-concurrent "${N_CONCURRENT}" \
  --no-delete \
  --agent-setup-timeout-multiplier 4.0 \
  --job-name "${JOB_NAME}"

# ============================================================
# 4. Extract patches from Harbor job
# ============================================================

mkdir -p "${RESULT_DIR}/patches"

uv run python -m hwe_bench.harness.harbor.verify_bridge \
  --harbor-job-dir "${JOB_DIR}" \
  --output "${RESULT_DIR}/patches"

# ============================================================
# 5. Evaluate all Ibex patches
# ============================================================

rm -rf "${RESULT_DIR}/eval_workdir"
rm -rf "${RESULT_DIR}/eval"
rm -rf "${RESULT_DIR}/eval_logs"

uv run python -m hwe_bench.harness.evaluator \
  --workdir "$(pwd)/${RESULT_DIR}/eval_workdir" \
  --patch_files "$(pwd)/${RESULT_DIR}/patches/patches.jsonl" \
  --dataset_files "$(pwd)/${DATASET_FILE}" \
  --output_dir "${RESULT_DIR}/eval" \
  --log_dir "$(pwd)/${RESULT_DIR}/eval_logs" \
  --stop_on_error false \
  --max_workers "${EVAL_WORKERS}" \
  --max_workers_build_image "${EVAL_WORKERS}" \
  --max_workers_run_instance "${EVAL_WORKERS}"