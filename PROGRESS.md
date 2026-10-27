# Progress & experiment log — adk-hybrid-triage

Updated 2026-10-27. Detailed per-run metrics: `results/experiment_log.csv`, `results/profile_comparison.csv`.

## Gate status

| Gate | gemini_pipeline | gemini_strong_only | local (qwen2.5:7b) |
|---|---|---|---|
| G0 pytest | 44/44 PASS (profile-independent) | ← | ← |
| G1 model smoke | fast OK; strong 3.8 Flash OK once on host (260 s), then 503/504 — **PENDING in Docker** | PENDING | PASS (LOCAL_MODEL_READY, 10.6 s cold, 100% CPU) |
| G2 preflight | GUARDRAIL pass; NORMAL/SECURITY = INFRA (3.8 Flash 503/504) | not run | PASS 3/3 |
| G3 guardrail | — | — | 100% (regex v2) |
| G4 security/P1 human review | — | — | 100% (6/6, run 4) |
| G5 tool behavior ≥95 | — | — | **87.9% FAIL** (skips lookup on simple questions) |
| G6 schema ≥95 | — | — | 100% (formatter step) |
| G7 category/priority/all | — | — | 96.2 / **65.4** / **60.6** (not blocking) |
| G8 reliability (3 runs same config) | — | — | 1 run of current config |
| G9 runtime HTTP | — | — | PASS with TRIAGE_PROFILE=local (deviation: Gemini unavailable) |

## Experiment log (one line each)

1. stage1 — Docker Desktop 4.96.3 (CLI 29.9.2, Compose v5.6.1); host Python 3.13.6.
2. stage2 — repo + .gitignore/.dockerignore/.env.example/MODEL_SELECTION.md; key only in .env (len 53).
3. stage3 — google-adk 2.13.0, google-genai 2.33.0, litellm 1.96.2; SDK default = no retries; 3.8 Flash slowness = server capacity (503), not thinking.
4. stage4 — py3.12.15 image, versions identical to host, .env absent from image; strong-model smoke pending (Google capacity).
5. stage5 — policy v1, tool, schemas, callbacks; added before_agent_guardrail at pipeline entry (deviation). 4 regex gaps found.
6. stage6 — 23 tests + 4 strict-xfail gaps → G0.
7. stage7 — 3 profiles; model IDs from env; Gemini retry 429/503 x3, 90 s timeout. SequentialAgent deprecated (keep; Workflow later).
8. stage11 — ollama 0.40.9 (Docker, CPU), qwen2.5:7b-instruct 845dbda0ea48, ctx 4096, VM 8.3 GB.
9. local run1/run2 (baseline) — critical 52.9/47.1%; junk-prefix JSON, 36-call tool loop (631 s).
10. local run3 — guardrail regex v2 (+17 holdout tests) → G3 100%; remaining critical fails all = output format.
11. hypothesis "context truncation" REJECTED (≤1005 tokens, truncated=0).
12. local run4 — formatter step (output_schema → constrained JSON) + loop cap 10 + 180 s local timeout → schema 100%, category 96.2%, G4 6/6; formatter changed 0/28 decisions (audited); latency median 25.7 s.
13. scorer — added guide-definition G4 (security/P1 HR) alongside stricter "critical" metric; all runs recomputed.
14. runtime — api_server hung on ADK telemetry prompt under TTY → fixed with `docker compose run -T`; HTTP smoke PASS (local).
15. hardening — ports bound to 127.0.0.1 (api_server has no auth); `.adk/` session DB git/docker-ignored.

## Open items (next session)

1. Retry 3.8 Flash: `gemini_sdk_smoke.py` in Docker → close Stage 4; then hosted preflight + evals (gemini_pipeline, gemini_strong_only) x3.
2. Local G5: tool skipped on how-to/invoice/holiday questions — one instruction change, measured.
3. G8: 3 repeated runs per candidate profile on the final config.
4. Priority calibration (UNKNOWN rarely used; over-escalation) — judgment work after G5.
5. compare_profiles → choose runtime profile by measured gates/latency; update MODEL_SELECTION.md decision.
6. Later: migrate SequentialAgent → Workflow; optional judge (judge_results.py, not yet written).
