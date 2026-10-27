# Model Selection Contract

Status: **HYPOTHESIS — not yet measured.** Written before any eval run (guide §7D).
Each row must be confirmed or overturned by the locked eval, not by model reputation.

Release rule: a cheaper/faster model is acceptable only if it clears the task's
accuracy threshold **and** every critical safety gate (100%). A stronger model is
kept only if the locked eval shows a measurable gain worth its cost/latency.

| Task | Default model | Why (hypothesis) | Metrics that could falsify it |
|---|---|---|---|
| Ticket fact extraction | `gemini-3.5-flash-lite` | Bounded, structured, high-volume normalisation → favour latency/cost. | Schema validity, extraction errors vs. ticket text, latency |
| Final triage + tool choice | `gemini-3.8-flash` | Priority, ambiguity, missing-info judgment and tool use need stronger reasoning. | All-fields accuracy, tool accuracy, critical gates, latency |
| Local benchmark / fallback | `qwen2.5:7b-instruct` (Ollama, CPU in Docker) | Measures privacy/control/cost trade-off vs hosted. | Same locked eval + local latency |
| Qualitative judge | `gemini-3.8-flash` | Grades groundedness / action safety outside runtime; never overrides hard gates. | Judge scores (caveat: same model as triage → self-preference risk) |

## Profiles to compare on the identical locked eval

- `gemini_pipeline` — Flash-Lite extractor → 3.8 Flash triage (default)
- `gemini_strong_only` — 3.8 Flash only (ablation: does the extractor earn its extra call?)
- `local` — Qwen2.5-7B via Ollama/LiteLLM

## Model IDs verified

- 2026-10-26: `gemini-3.5-flash-lite` and `gemini-3.8-flash` listed on
  ai.google.dev/gemini-api/docs/deprecations with no shutdown date.
- 2026-10-26: `qwen2.5:7b-instruct` exists on ollama.com (4.7 GB, Q4_K_M).

## Deviation: temporary strong model (2026-10-27)

- `gemini-3.8-flash` returned 503 / 504 on every attempt for ~24 h (host and Docker; 1 success
  in 260 s on 2026-10-26). Other free-tier users report the same on the Gemini forum.
- `gemini-3.6-flash` (Google stable, no shutdown date) passed the SDK smoke test: exact match, 2.14 s.
- Hosted evals therefore run with `GEMINI_STRONG_MODEL=gemini-3.6-flash` passed via `-e` per command;
  `.env` and the contract above are unchanged. The experiment log records the actual model per run.
- 3.8 Flash must still be evaluated on the same locked set before any final decision that names it.

## Decision (fill in after Stage 12)

_Pending measured results._
