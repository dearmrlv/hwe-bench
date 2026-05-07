# HWE Eval Run Queue

## Completed

- Finish the current ibex 35-case `gen-patch` run:
  - design: `ibex`
  - dataset: `datasets/lowRISC__ibex.jsonl`
  - agent: `openhands`
  - model: `gpt-5.5`
  - base URL: `https://sub2api.llm.icbench.com/v1`
  - run directory: `results/ibex/openhands/gpt-5.5/ibex_35_gpt55_openhands/`
- Result: generation completed for 35/35 cases.
- Fixup: `ibex-pr-882` initially produced an empty patch due to an OpenHands/LiteLLM provider error; reran that case with explicit provider prefix and merged the non-empty patch into the main job.
- Eval result: 35 submitted, 35 completed, 30 resolved, 5 unresolved, 0 empty patches, 0 eval errors.

## Active

- Initialize and run the queued non-ibex designs below. Do not store API keys in repo files.

For each design below, first test one case with:

- agent/model flow: one-case smoke test
- model: `qwen3.6-plus`
- base URL: `https://coding.dashscope.aliyuncs.com/v1`
- API key: pass at runtime only; do not store in repo files

Then launch all cases with:

- agent: `openhands`
- model: `gpt-5.5`
- base URL: `https://sub2api.llm.icbench.com/v1`
- API key: pass at runtime only; do not store in repo files

Design queue:

- `cva6`
- `caliptra-rtl`
- `rocket-chip`
- `XiangShan`
