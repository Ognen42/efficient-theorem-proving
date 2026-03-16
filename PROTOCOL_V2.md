# Kimina Eval Protocol V2

This document defines the canonical evaluation protocol used in this repository.

## Version
- `protocol_name`: `kimina_eval_v2`
- `protocol_version`: `2.0.0`

## Fixed defaults
- `pass@k`: `k = 4`
- Kimina verification timeout: `60` seconds
- Temperature: `0.7`
- Top-p: `0.95`
- Max generation tokens: `8096`
- Samples per generation call: `4`

## Prompt contract
- System role: `You are an expert in mathematics and Lean 4.`
- User message includes:
  - Solve step by step in Lean 4.
  - Explicit final-answer instruction:
    `After reasoning, provide one final proof attempt in Lean 4 as a single ```lean4``` block, introduced by the line: Here is the final proof:`
  - Formal statement in a `lean4` code fence.

## Generation and extraction
- Generate up to `k=4` attempts per problem.
- Extract the **last** ` ```lean4 ... ``` ` block from model output.
- Proof extraction: content after first `:=` if present, else full block.

## Verification snippet assembly
- Prefix snippet with:
  - `import Mathlib`
  - `import Aesop`
  - `set_option maxHeartbeats 0`
  - `open BigOperators Real Nat Topology Rat`
- Attach proof robustly using statement-aware composition (do not use blind string slicing).

## Verification criterion
- Call Kimina `/check` with `timeout=60`.
- A sample is valid iff `analysis.status.value == "valid"`.
- A problem is solved at pass@k iff at least one of the first `k` attempts is valid.

## Persistence contract
- Save per-run metadata with:
  - protocol name/version
  - model path
  - dataset path
  - generation settings
  - verification timeout
  - timestamp
- Keep archived pre-v2 runs labeled as `v1_archive` and do not mix with v2 in aggregate comparisons.
