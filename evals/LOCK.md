# Eval set lock — triage_cases_v1

- File: `evals/triage_cases.jsonl`
- Cases: 33 (17 critical)
- SHA-256: `5f563dcd34b244be44c5ebcbe5832bb31627f14e79eb043b4d5a7fd27d7639ca`
- Locked: 2026-10-26, against `policy.py` POLICY_VERSION = v1
- Written from the policy before any eval run. Guide's six examples copied verbatim.

## Rules
1. Never edit v1. Any change → new file `triage_cases_v2.jsonl`, new hash, and all
   profiles are re-run on v2. Results across versions are not compared.
2. Never relabel a case because a model failed it. If a label is genuinely wrong,
   record why in this file *before* creating v2.
3. `ambiguous` cases carry acceptable sets (e.g. priority P2|P3). Critical rules
   (security / P1 → human_review, guardrail block/allow) are exact.
4. `known_gap` cases (4) are expected to fail with guardrail regex v1. They stay in
   the set: they are why G3 is not yet 100%, and they will prove the fix.

## Pre-run check (no model): current regex vs expected guardrail_block
Mismatches = exactly the 4 known gaps: gap_password_colon, gap_benign_system_prompt,
gap_instruction_exfil, gap_bare_google_key.
